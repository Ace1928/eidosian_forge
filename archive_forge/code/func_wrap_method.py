import enum
import functools
from google.api_core import grpc_helpers
from google.api_core.gapic_v1 import client_info
from google.api_core.timeout import TimeToDeadlineTimeout
def wrap_method(func, default_retry=None, default_timeout=None, default_compression=None, client_info=client_info.DEFAULT_CLIENT_INFO, *, with_call=False):
    """Wrap an RPC method with common behavior.

    This applies common error wrapping, retry, timeout, and compression behavior to a function.
    The wrapped function will take optional ``retry``, ``timeout``, and ``compression``
    arguments.

    For example::

        import google.api_core.gapic_v1.method
        from google.api_core import retry
        from google.api_core import timeout
        from grpc import Compression

        # The original RPC method.
        def get_topic(name, timeout=None):
            request = publisher_v2.GetTopicRequest(name=name)
            return publisher_stub.GetTopic(request, timeout=timeout)

        default_retry = retry.Retry(deadline=60)
        default_timeout = timeout.Timeout(deadline=60)
        default_compression = Compression.NoCompression
        wrapped_get_topic = google.api_core.gapic_v1.method.wrap_method(
            get_topic, default_retry)

        # Execute get_topic with default retry and timeout:
        response = wrapped_get_topic()

        # Execute get_topic without doing any retying but with the default
        # timeout:
        response = wrapped_get_topic(retry=None)

        # Execute get_topic but only retry on 5xx errors:
        my_retry = retry.Retry(retry.if_exception_type(
            exceptions.InternalServerError))
        response = wrapped_get_topic(retry=my_retry)

    The way this works is by late-wrapping the given function with the retry
    and timeout decorators. Essentially, when ``wrapped_get_topic()`` is
    called:

    * ``get_topic()`` is first wrapped with the ``timeout`` into
      ``get_topic_with_timeout``.
    * ``get_topic_with_timeout`` is wrapped with the ``retry`` into
      ``get_topic_with_timeout_and_retry()``.
    * The final ``get_topic_with_timeout_and_retry`` is called passing through
      the ``args``  and ``kwargs``.

    The callstack is therefore::

        method.__call__() ->
            Retry.__call__() ->
                Timeout.__call__() ->
                    wrap_errors() ->
                        get_topic()

    Note that if ``timeout`` or ``retry`` is ``None``, then they are not
    applied to the function. For example,
    ``wrapped_get_topic(timeout=None, retry=None)`` is more or less
    equivalent to just calling ``get_topic`` but with error re-mapping.

    Args:
        func (Callable[Any]): The function to wrap. It should accept an
            optional ``timeout`` argument. If ``metadata`` is not ``None``, it
            should accept a ``metadata`` argument.
        default_retry (Optional[google.api_core.Retry]): The default retry
            strategy. If ``None``, the method will not retry by default.
        default_timeout (Optional[google.api_core.Timeout]): The default
            timeout strategy. Can also be specified as an int or float. If
            ``None``, the method will not have timeout specified by default.
        default_compression (Optional[grpc.Compression]): The default
            grpc.Compression. If ``None``, the method will not have
            compression specified by default.
        client_info
            (Optional[google.api_core.gapic_v1.client_info.ClientInfo]):
                Client information used to create a user-agent string that's
                passed as gRPC metadata to the method. If unspecified, then
                a sane default will be used. If ``None``, then no user agent
                metadata will be provided to the RPC method.
        with_call (bool): If True, wrapped grpc.UnaryUnaryMulticallables will
            return a tuple of (response, grpc.Call) instead of just the response.
            This is useful for extracting trailing metadata from unary calls.
            Defaults to False.

    Returns:
        Callable: A new callable that takes optional ``retry``, ``timeout``,
            and ``compression``
            arguments and applies the common error mapping, retry, timeout, compression,
            and metadata behavior to the low-level RPC method.
    """
    if with_call:
        try:
            func = func.with_call
        except AttributeError as exc:
            raise ValueError('with_call=True is only supported for unary calls.') from exc
    func = grpc_helpers.wrap_errors(func)
    if client_info is not None:
        user_agent_metadata = [client_info.to_grpc_metadata()]
    else:
        user_agent_metadata = None
    return functools.wraps(func)(_GapicCallable(func, default_retry, default_timeout, default_compression, metadata=user_agent_metadata))