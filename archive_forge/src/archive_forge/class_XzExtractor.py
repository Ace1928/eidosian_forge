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
class XzExtractor(MagicNumberBaseExtractor):
    magic_numbers = [b'\xfd7zXZ\x00']

    @staticmethod
    def extract(input_path: Union[Path, str], output_path: Union[Path, str]) -> None:
        with lzma.open(input_path) as compressed_file:
            with open(output_path, 'wb') as extracted_file:
                shutil.copyfileobj(compressed_file, extracted_file)