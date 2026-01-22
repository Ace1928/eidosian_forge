import asyncio
import functools
import logging
import time
from inspect import iscoroutinefunction, isfunction
from typing import Any, Callable, Dict, Tuple, TypeVar

# Type variable for the decorator
F = TypeVar("F", bound=Callable[..., Any])


def standard_decorator(
    retries: int = 3,
    delay: int = 2,
    cache_results: bool = False,
    log_performance: bool = True,
):
    """
    A decorator that enhances a function with comprehensive features including logging, error handling,
    performance monitoring, automatic retrying on transient failures, and optional result caching.
    It is designed to be flexible and applicable to both synchronous and asynchronous functions.

    Args:
        retries (int): Number of times to retry the function on failure.
        delay (int): Delay between retries in seconds.
        cache_results (bool): Whether to cache the function's return value.
        log_performance (bool): Whether to log the performance of the function.

    Returns:
        Callable[..., Any]: A wrapped function with enhanced capabilities.
    """

    def decorator(func: F) -> F:
        cache: Dict[Tuple, Any] = {}

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal cache
            key = (args, frozenset(kwargs.items()))
            if cache_results and key in cache:
                logging.debug(
                    f"Cache hit for {func.__name__} with args: {args}, kwargs: {kwargs}"
                )
                return cache[key]

            attempts = 0
            while attempts < retries:
                try:
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    end_time = time.time()
                    if log_performance:
                        logging.debug(
                            f"{func.__name__} executed in {end_time - start_time:.2f}s"
                        )
                    if cache_results:
                        cache[key] = result
                    return result
                except Exception as e:
                    logging.error(
                        f"{func.__name__} attempt {attempts + 1} failed with {e}, retrying..."
                    )
                    attempts += 1
                    await asyncio.sleep(delay)
            logging.debug(f"Final attempt for {func.__name__}")
            return await func(
                *args, **kwargs
            )  # Final attempt without catching exceptions

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            nonlocal cache
            key = (args, frozenset(kwargs.items()))
            if cache_results and key in cache:
                logging.debug(
                    f"Cache hit for {func.__name__} with args: {args}, kwargs: {kwargs}"
                )
                return cache[key]

            attempts = 0
            while attempts < retries:
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    if log_performance:
                        logging.debug(
                            f"{func.__name__} executed in {end_time - start_time:.2f}s"
                        )
                    if cache_results:
                        cache[key] = result
                    return result
                except Exception as e:
                    logging.error(
                        f"{func.__name__} attempt {attempts + 1} failed with {e}, retrying..."
                    )
                    attempts += 1
                    time.sleep(delay)
            logging.debug(f"Final attempt for {func.__name__}")
            return func(*args, **kwargs)  # Final attempt without catching exceptions

        if iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator
