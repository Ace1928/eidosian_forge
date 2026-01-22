import asyncio
import functools
import json
import random
import re
import sys
import zlib
from enum import IntEnum
from struct import Struct
from typing import (
from .base_protocol import BaseProtocol
from .compression_utils import ZLibCompressor, ZLibDecompressor
from .helpers import NO_EXTENSIONS
from .streams import DataQueue
def _make_compress_obj(self, compress: int) -> ZLibCompressor:
    return ZLibCompressor(level=zlib.Z_BEST_SPEED, wbits=-compress, max_sync_chunk_size=WEBSOCKET_MAX_SYNC_CHUNK_SIZE)