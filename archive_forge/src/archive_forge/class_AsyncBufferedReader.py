from ..base import AsyncBase, AsyncIndirectBase
from .utils import delegate_to_executor, proxy_method_directly, proxy_property_directly
@delegate_to_executor('peek')
class AsyncBufferedReader(AsyncBufferedIOBase):
    """The asyncio executor version of io.BufferedReader and Random."""