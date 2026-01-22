from __future__ import division
import array
import os
import subprocess
from tempfile import TemporaryFile, NamedTemporaryFile
import wave
import sys
import struct
from .logging_utils import log_conversion, log_subprocess_output
from .utils import mediainfo_json, fsdecode
import base64
from collections import namedtuple
from io import BytesIO
from .utils import (
from .exceptions import (
from . import effects
@classmethod
def _from_safe_wav(cls, file):
    file, close_file = _fd_or_path_or_tempfile(file, 'rb', tempfile=False)
    file.seek(0)
    obj = cls(data=file)
    if close_file:
        file.close()
    return obj