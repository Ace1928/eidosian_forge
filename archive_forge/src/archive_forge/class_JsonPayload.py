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
class JsonPayload(BytesPayload):

    def __init__(self, value: Any, encoding: str='utf-8', content_type: str='application/json', dumps: JSONEncoder=json.dumps, *args: Any, **kwargs: Any) -> None:
        super().__init__(dumps(value).encode(encoding), *args, content_type=content_type, encoding=encoding, **kwargs)