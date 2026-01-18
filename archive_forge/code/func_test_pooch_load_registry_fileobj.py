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
def test_pooch_load_registry_fileobj():
    """Loading the registry from a file object"""
    path = os.path.join(DATA_DIR, 'registry.txt')
    pup = Pooch(path='', base_url='')
    with open(path, 'rb') as fin:
        pup.load_registry(fin)
    assert pup.registry == REGISTRY
    assert pup.registry_files.sort() == list(REGISTRY).sort()
    pup = Pooch(path='', base_url='')
    with open(path, 'r', encoding='utf-8') as fin:
        pup.load_registry(fin)
    assert pup.registry == REGISTRY
    assert pup.registry_files.sort() == list(REGISTRY).sort()