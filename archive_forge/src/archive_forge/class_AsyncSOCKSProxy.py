from .connection import AsyncHTTPConnection
from .connection_pool import AsyncConnectionPool
from .http11 import AsyncHTTP11Connection
from .http_proxy import AsyncHTTPProxy
from .interfaces import AsyncConnectionInterface
class AsyncSOCKSProxy:

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("Attempted to use SOCKS support, but the `socksio` package is not installed. Use 'pip install httpcore[socks]'.")