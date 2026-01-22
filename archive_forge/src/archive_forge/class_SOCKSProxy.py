from .connection import HTTPConnection
from .connection_pool import ConnectionPool
from .http11 import HTTP11Connection
from .http_proxy import HTTPProxy
from .interfaces import ConnectionInterface
class SOCKSProxy:

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("Attempted to use SOCKS support, but the `socksio` package is not installed. Use 'pip install httpcore[socks]'.")