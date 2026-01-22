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
class QuerystringState(IntEnum):
    """Querystring parser states.

    These are used to keep track of the state of the parser, and are used to determine
    what to do when new data is encountered.
    """
    BEFORE_FIELD = 0
    FIELD_NAME = 1
    FIELD_DATA = 2