import sys
from abc import ABC
from asyncio import IncompleteReadError, StreamReader, TimeoutError
from typing import List, Optional, Union
from ..exceptions import (
from ..typing import EncodableT
from .encoders import Encoder
from .socket import SERVER_CLOSED_CONNECTION_ERROR, SocketBuffer
class AsyncBaseParser(BaseParser):
    """Base parsing class for the python-backed async parser"""
    __slots__ = ('_stream', '_read_size')

    def __init__(self, socket_read_size: int):
        self._stream: Optional[StreamReader] = None
        self._read_size = socket_read_size

    async def can_read_destructive(self) -> bool:
        raise NotImplementedError()

    async def read_response(self, disable_decoding: bool=False) -> Union[EncodableT, ResponseError, None, List[EncodableT]]:
        raise NotImplementedError()