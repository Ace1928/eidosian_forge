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
class GzipExtractor(MagicNumberBaseExtractor):
    magic_numbers = [b'\x1f\x8b']

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        with gzip.open(input_path, 'rb') as gzip_file:
            with open(output_path, 'wb') as extracted_file:
                shutil.copyfileobj(gzip_file, extracted_file)