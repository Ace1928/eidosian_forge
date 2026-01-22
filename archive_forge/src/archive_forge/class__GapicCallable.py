import enum
import functools
from google.api_core import grpc_helpers
from google.api_core.gapic_v1 import client_info
from google.api_core.timeout import TimeToDeadlineTimeout
class _GapicCallable(object):
    """Callable that applies retry, timeout, and metadata logic.

    Args:
        target (Callable): The low-level RPC method.
        retry (google.api_core.retry.Retry): The default retry for the
            callable. If ``None``, this callable will not retry by default
        timeout (google.api_core.timeout.Timeout): The default timeout for the
            callable (i.e. duration of time within which an RPC must terminate
            after its start, not to be confused with deadline). If ``None``,
            this callable will not specify a timeout argument to the low-level
            RPC method.
        compression (grpc.Compression): The default compression for the callable.
            If ``None``, this callable will not specify a compression argument
            to the low-level RPC method.
        metadata (Sequence[Tuple[str, str]]): Additional metadata that is
            provided to the RPC method on every invocation. This is merged with
            any metadata specified during invocation. If ``None``, no
            additional metadata will be passed to the RPC method.
    """

    def __init__(self, target, retry, timeout, compression, metadata=None):
        self._target = target
        self._retry = retry
        self._timeout = timeout
        self._compression = compression
        self._metadata = metadata

    def __call__(self, *args, timeout=DEFAULT, retry=DEFAULT, compression=DEFAULT, **kwargs):
        """Invoke the low-level RPC with retry, timeout, compression, and metadata."""
        if retry is DEFAULT:
            retry = self._retry
        if timeout is DEFAULT:
            timeout = self._timeout
        if compression is DEFAULT:
            compression = self._compression
        if isinstance(timeout, (int, float)):
            timeout = TimeToDeadlineTimeout(timeout=timeout)
        wrapped_func = _apply_decorators(self._target, [retry, timeout])
        if self._metadata is not None:
            metadata = kwargs.get('metadata', [])
            if metadata is None:
                metadata = []
            metadata = list(metadata)
            metadata.extend(self._metadata)
            kwargs['metadata'] = metadata
        if self._compression is not None:
            kwargs['compression'] = compression
        return wrapped_func(*args, **kwargs)