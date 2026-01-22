from __future__ import annotations
import logging
import os
import shutil
import sys
import tempfile
from email.message import Message
from enum import IntEnum
from io import BytesIO
from numbers import Number
from typing import TYPE_CHECKING
from .decoders import Base64Decoder, QuotedPrintableDecoder
from .exceptions import FileError, FormParserError, MultipartParseError, QuerystringParseError
class QuerystringCallbacks(TypedDict, total=False):
    on_field_start: Callable[[], None]
    on_field_name: Callable[[bytes, int, int], None]
    on_field_data: Callable[[bytes, int, int], None]
    on_field_end: Callable[[], None]
    on_end: Callable[[], None]