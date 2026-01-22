from ..base import AsyncBase, AsyncIndirectBase
from .utils import delegate_to_executor, proxy_method_directly, proxy_property_directly
@delegate_to_executor('close', 'flush', 'isatty', 'read', 'read1', 'readinto', 'readline', 'readlines', 'seek', 'seekable', 'tell', 'truncate', 'writable', 'write', 'writelines')
@proxy_method_directly('detach', 'fileno', 'readable')
@proxy_property_directly('closed', 'raw', 'name', 'mode')
class AsyncBufferedIOBase(AsyncBase):
    """The asyncio executor version of io.BufferedWriter and BufferedIOBase."""