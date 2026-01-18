import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from ..core import create, Pooch, retrieve, download_action, stream_download
from ..utils import get_logger, temporary_file, os_cache
from ..hashes import file_hash, hash_matches
from .. import core
from ..downloaders import HTTPDownloader, FTPDownloader
from .utils import (
def test_pooch_load_registry_invalid_line():
    """Should raise an exception when a line doesn't have two elements"""
    pup = Pooch(path='', base_url='', registry={})
    with pytest.raises(IOError):
        pup.load_registry(os.path.join(DATA_DIR, 'registry-invalid.txt'))