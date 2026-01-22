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
class MultipartState(IntEnum):
    """Multipart parser states.

    These are used to keep track of the state of the parser, and are used to determine
    what to do when new data is encountered.
    """
    START = 0
    START_BOUNDARY = 1
    HEADER_FIELD_START = 2
    HEADER_FIELD = 3
    HEADER_VALUE_START = 4
    HEADER_VALUE = 5
    HEADER_VALUE_ALMOST_DONE = 6
    HEADERS_ALMOST_DONE = 7
    PART_DATA_START = 8
    PART_DATA = 9
    PART_DATA_END = 10
    END = 11