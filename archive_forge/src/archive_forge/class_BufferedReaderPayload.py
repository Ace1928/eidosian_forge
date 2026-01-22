import asyncio
import enum
import io
import json
import mimetypes
import os
import warnings
from abc import ABC, abstractmethod
from itertools import chain
from typing import (
from multidict import CIMultiDict
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import (
from .streams import StreamReader
from .typedefs import JSONEncoder, _CIMultiDict
class BufferedReaderPayload(IOBasePayload):

    @property
    def size(self) -> Optional[int]:
        try:
            return os.fstat(self._value.fileno()).st_size - self._value.tell()
        except OSError:
            return None