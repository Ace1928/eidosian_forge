import base64
import binascii
import json
import re
import uuid
import warnings
import zlib
from collections import deque
from types import TracebackType
from typing import (
from urllib.parse import parse_qsl, unquote, urlencode
from multidict import CIMultiDict, CIMultiDictProxy, MultiMapping
from .compression_utils import ZLibCompressor, ZLibDecompressor
from .hdrs import (
from .helpers import CHAR, TOKEN, parse_mimetype, reify
from .http import HeadersParser
from .payload import (
from .streams import StreamReader
@payload_type(BodyPartReader, order=Order.try_first)
class BodyPartReaderPayload(Payload):

    def __init__(self, value: BodyPartReader, *args: Any, **kwargs: Any) -> None:
        super().__init__(value, *args, **kwargs)
        params: Dict[str, str] = {}
        if value.name is not None:
            params['name'] = value.name
        if value.filename is not None:
            params['filename'] = value.filename
        if params:
            self.set_content_disposition('attachment', True, **params)

    async def write(self, writer: Any) -> None:
        field = self._value
        chunk = await field.read_chunk(size=2 ** 16)
        while chunk:
            await writer.write(field.decode(chunk))
            chunk = await field.read_chunk(size=2 ** 16)