import bz2
import gzip
import lzma
import os
import shutil
import struct
import tarfile
import warnings
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type, Union
from .. import config
from ._filelock import FileLock
from .logging import get_logger
class MagicNumberBaseExtractor(BaseExtractor, ABC):
    magic_numbers: List[bytes] = []

    @staticmethod
    def read_magic_number(path: Union[Path, str], magic_number_length: int):
        with open(path, 'rb') as f:
            return f.read(magic_number_length)

    @classmethod
    def is_extractable(cls, path: Union[Path, str], magic_number: bytes=b'') -> bool:
        if not magic_number:
            magic_number_length = max((len(cls_magic_number) for cls_magic_number in cls.magic_numbers))
            try:
                magic_number = cls.read_magic_number(path, magic_number_length)
            except OSError:
                return False
        return any((magic_number.startswith(cls_magic_number) for cls_magic_number in cls.magic_numbers))