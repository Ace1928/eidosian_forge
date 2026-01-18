import asyncio
import mimetypes
import os
import pathlib
from typing import (  # noqa
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import ETAG_ANY, ETag, must_be_empty_body
from .typedefs import LooseHeaders, PathLike
from .web_exceptions import (
from .web_response import StreamResponse
Return the file path, stat result, and gzip status.

        This method should be called from a thread executor
        since it calls os.stat which may block.
        