import os
import re
import sys
from typing import BinaryIO, Optional, Tuple, Union
import torch
import torchaudio
from .backend import Backend
from .common import AudioMetaData
def _native_endianness() -> str:
    if sys.byteorder == 'little':
        return 'le'
    else:
        return 'be'