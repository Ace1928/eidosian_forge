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
class Extractor:
    extractors: Dict[str, Type[BaseExtractor]] = {'tar': TarExtractor, 'gzip': GzipExtractor, 'zip': ZipExtractor, 'xz': XzExtractor, 'rar': RarExtractor, 'zstd': ZstdExtractor, 'bz2': Bzip2Extractor, '7z': SevenZipExtractor, 'lz4': Lz4Extractor}

    @classmethod
    def _get_magic_number_max_length(cls):
        return max((len(extractor_magic_number) for extractor in cls.extractors.values() if issubclass(extractor, MagicNumberBaseExtractor) for extractor_magic_number in extractor.magic_numbers))

    @staticmethod
    def _read_magic_number(path: Union[Path, str], magic_number_length: int):
        try:
            return MagicNumberBaseExtractor.read_magic_number(path, magic_number_length=magic_number_length)
        except OSError:
            return b''

    @classmethod
    def is_extractable(cls, path: Union[Path, str], return_extractor: bool=False) -> bool:
        warnings.warn("Method 'is_extractable' was deprecated in version 2.4.0 and will be removed in 3.0.0. Use 'infer_extractor_format' instead.", category=FutureWarning)
        extractor_format = cls.infer_extractor_format(path)
        if extractor_format:
            return True if not return_extractor else (True, cls.extractors[extractor_format])
        return False if not return_extractor else (False, None)

    @classmethod
    def infer_extractor_format(cls, path: Union[Path, str]) -> str:
        magic_number_max_length = cls._get_magic_number_max_length()
        magic_number = cls._read_magic_number(path, magic_number_max_length)
        for extractor_format, extractor in cls.extractors.items():
            if extractor.is_extractable(path, magic_number=magic_number):
                return extractor_format

    @classmethod
    def extract(cls, input_path: Union[Path, str], output_path: Union[Path, str], extractor_format: Optional[str]=None, extractor: Optional[BaseExtractor]='deprecated') -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lock_path = str(Path(output_path).with_suffix('.lock'))
        with FileLock(lock_path):
            shutil.rmtree(output_path, ignore_errors=True)
            if extractor_format or extractor != 'deprecated':
                if extractor != 'deprecated' or not isinstance(extractor_format, str):
                    warnings.warn("Parameter 'extractor' was deprecated in version 2.4.0 and will be removed in 3.0.0. Use 'extractor_format' instead.", category=FutureWarning)
                    extractor = extractor if extractor != 'deprecated' else extractor_format
                else:
                    extractor = cls.extractors[extractor_format]
                return extractor.extract(input_path, output_path)
            else:
                warnings.warn("Parameter 'extractor_format' was made required in version 2.4.0 and not passing it will raise an exception in 3.0.0.", category=FutureWarning)
                for extractor in cls.extractors.values():
                    if extractor.is_extractable(input_path):
                        return extractor.extract(input_path, output_path)