import functools
from google.api_core import grpc_helpers_async
from google.api_core.gapic_v1 import client_info
from google.api_core.gapic_v1.method import _GapicCallable
from google.api_core.gapic_v1.method import DEFAULT  # noqa: F401
from google.api_core.gapic_v1.method import USE_DEFAULT_METADATA  # noqa: F401
Wrap an async RPC method with common behavior.

    Returns:
        Callable: A new callable that takes optional ``retry``, ``timeout``,
            and ``compression`` arguments and applies the common error mapping,
            retry, timeout, metadata, and compression behavior to the low-level RPC method.
    