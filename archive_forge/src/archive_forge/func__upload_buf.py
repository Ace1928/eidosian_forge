import hashlib
import io
import json
import os
import platform
import random
import socket
import ssl
import threading
import time
import urllib.parse
from typing import (
import filelock
import urllib3
from blobfile import _xml as xml
def _upload_buf(self, buf: memoryview, finalize: bool=False) -> int:
    if finalize:
        size = len(buf)
    else:
        size = len(buf) // self._chunk_size * self._chunk_size
        assert size > 0
    chunk = buf[:size]
    self._upload_chunk(chunk, finalize)
    self._offset += len(chunk)
    return size