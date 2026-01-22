import io
from typing import Any, Iterable, List, Optional
from urllib.parse import urlencode
from multidict import MultiDict, MultiDictProxy
from . import hdrs, multipart, payload
from .helpers import guess_filename
from .payload import Payload
def add_field(self, name: str, value: Any, *, content_type: Optional[str]=None, filename: Optional[str]=None, content_transfer_encoding: Optional[str]=None) -> None:
    if isinstance(value, io.IOBase):
        self._is_multipart = True
    elif isinstance(value, (bytes, bytearray, memoryview)):
        if filename is None and content_transfer_encoding is None:
            filename = name
    type_options: MultiDict[str] = MultiDict({'name': name})
    if filename is not None and (not isinstance(filename, str)):
        raise TypeError('filename must be an instance of str. Got: %s' % filename)
    if filename is None and isinstance(value, io.IOBase):
        filename = guess_filename(value, name)
    if filename is not None:
        type_options['filename'] = filename
        self._is_multipart = True
    headers = {}
    if content_type is not None:
        if not isinstance(content_type, str):
            raise TypeError('content_type must be an instance of str. Got: %s' % content_type)
        headers[hdrs.CONTENT_TYPE] = content_type
        self._is_multipart = True
    if content_transfer_encoding is not None:
        if not isinstance(content_transfer_encoding, str):
            raise TypeError('content_transfer_encoding must be an instance of str. Got: %s' % content_transfer_encoding)
        headers[hdrs.CONTENT_TRANSFER_ENCODING] = content_transfer_encoding
        self._is_multipart = True
    self._fields.append((type_options, headers, value))