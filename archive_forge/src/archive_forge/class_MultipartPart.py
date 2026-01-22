from __future__ import annotations
import typing
from dataclasses import dataclass, field
from enum import Enum
from tempfile import SpooledTemporaryFile
from urllib.parse import unquote_plus
from starlette.datastructures import FormData, Headers, UploadFile
@dataclass
class MultipartPart:
    content_disposition: bytes | None = None
    field_name: str = ''
    data: bytes = b''
    file: UploadFile | None = None
    item_headers: list[tuple[bytes, bytes]] = field(default_factory=list)