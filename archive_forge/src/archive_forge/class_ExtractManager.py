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
class ExtractManager:

    def __init__(self, cache_dir: Optional[str]=None):
        self.extract_dir = os.path.join(cache_dir, config.EXTRACTED_DATASETS_DIR) if cache_dir else config.EXTRACTED_DATASETS_PATH
        self.extractor = Extractor

    def _get_output_path(self, path: str) -> str:
        from .file_utils import hash_url_to_filename
        abs_path = os.path.abspath(path)
        return os.path.join(self.extract_dir, hash_url_to_filename(abs_path))

    def _do_extract(self, output_path: str, force_extract: bool) -> bool:
        return force_extract or (not os.path.isfile(output_path) and (not (os.path.isdir(output_path) and os.listdir(output_path))))

    def extract(self, input_path: str, force_extract: bool=False) -> str:
        extractor_format = self.extractor.infer_extractor_format(input_path)
        if not extractor_format:
            return input_path
        output_path = self._get_output_path(input_path)
        if self._do_extract(output_path, force_extract):
            self.extractor.extract(input_path, output_path, extractor_format)
        return output_path