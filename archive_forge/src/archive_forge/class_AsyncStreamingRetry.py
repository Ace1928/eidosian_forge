from __future__ import annotations
from typing import (
import asyncio
import time
import sys
import functools
from google.api_core.retry.retry_base import _BaseRetry
from google.api_core.retry.retry_base import _retry_error_helper
from google.api_core.retry import exponential_sleep_generator
from google.api_core.retry import build_retry_error
from google.api_core.retry import RetryFailureReason
class AsyncStreamingRetry(_BaseRetry):
    """Exponential retry decorator for async streaming rpcs.

    This class returns an AsyncGenerator when called, which wraps the target
    stream in retry logic. If any exception is raised by the target, the
    entire stream will be retried within the wrapper.

    Although the default behavior is to retry transient API errors, a
    different predicate can be provided to retry other exceptions.

    Important Note: when a stream is encounters a retryable error, it will
    silently construct a fresh iterator instance in the background
    and continue yielding (likely duplicate) values as if no error occurred.
    This is the most general way to retry a stream, but it often is not the
    desired behavior. Example: iter([1, 2, 1/0]) -> [1, 2, 1, 2, ...]

    There are two ways to build more advanced retry logic for streams:

    1. Wrap the target
        Use a ``target`` that maintains state between retries, and creates a
        different generator on each retry call. For example, you can wrap a
        grpc call in a function that modifies the request based on what has
        already been returned:

        .. code-block:: python

            async def attempt_with_modified_request(target, request, seen_items=[]):
                # remove seen items from request on each attempt
                new_request = modify_request(request, seen_items)
                new_generator = await target(new_request)
                async for item in new_generator:
                    yield item
                    seen_items.append(item)

            retry_wrapped = AsyncRetry(is_stream=True,...)(attempt_with_modified_request, target, request, [])

        2. Wrap the retry generator
            Alternatively, you can wrap the retryable generator itself before
            passing it to the end-user to add a filter on the stream. For
            example, you can keep track of the items that were successfully yielded
            in previous retry attempts, and only yield new items when the
            new attempt surpasses the previous ones:

            .. code-block:: python

                async def retryable_with_filter(target):
                    stream_idx = 0
                    # reset stream_idx when the stream is retried
                    def on_error(e):
                        nonlocal stream_idx
                        stream_idx = 0
                    # build retryable
                    retryable_gen = AsyncRetry(is_stream=True, ...)(target)
                    # keep track of what has been yielded out of filter
                    seen_items = []
                    async for item in retryable_gen:
                        if stream_idx >= len(seen_items):
                            yield item
                            seen_items.append(item)
                        elif item != previous_stream[stream_idx]:
                            raise ValueError("Stream differs from last attempt")"
                        stream_idx += 1

                filter_retry_wrapped = retryable_with_filter(target)

    Args:
        predicate (Callable[Exception]): A callable that should return ``True``
            if the given exception is retryable.
        initial (float): The minimum amount of time to delay in seconds. This
            must be greater than 0.
        maximum (float): The maximum amount of time to delay in seconds.
        multiplier (float): The multiplier applied to the delay.
        timeout (Optional[float]): How long to keep retrying in seconds.
            Note: timeout is only checked before initiating a retry, so the target may
            run past the timeout value as long as it is healthy.
        on_error (Optional[Callable[Exception]]): A function to call while processing
            a retryable exception. Any error raised by this function will
            *not* be caught.
        is_stream (bool): Indicates whether the input function
            should be treated as a stream function (i.e. an AsyncGenerator,
            or function or coroutine that returns an AsyncIterable).
            If True, the iterable will be wrapped with retry logic, and any
            failed outputs will restart the stream. If False, only the input
            function call itself will be retried. Defaults to False.
            To avoid duplicate values, retryable streams should typically be
            wrapped in additional filter logic before use.
        deadline (float): DEPRECATED use ``timeout`` instead. If set it will
        override ``timeout`` parameter.
    """

    def __call__(self, func: Callable[..., AsyncIterable[_Y] | Awaitable[AsyncIterable[_Y]]], on_error: Callable[[Exception], Any] | None=None) -> Callable[_P, Awaitable[AsyncGenerator[_Y, None]]]:
        """Wrap a callable with retry behavior.

        Args:
            func (Callable): The callable or stream to add retry behavior to.
            on_error (Optional[Callable[Exception]]): If given, the
                on_error callback will be called with each retryable exception
                raised by the wrapped function. Any error raised by this
                function will *not* be caught. If on_error was specified in the
                constructor, this value will be ignored.

        Returns:
            Callable: A callable that will invoke ``func`` with retry
                behavior.
        """
        if self._on_error is not None:
            on_error = self._on_error

        @functools.wraps(func)
        async def retry_wrapped_func(*args: _P.args, **kwargs: _P.kwargs) -> AsyncGenerator[_Y, None]:
            """A wrapper that calls target function with retry."""
            sleep_generator = exponential_sleep_generator(self._initial, self._maximum, multiplier=self._multiplier)
            return retry_target_stream(func, self._predicate, sleep_generator, self._timeout, on_error, init_args=args, init_kwargs=kwargs)
        return retry_wrapped_func