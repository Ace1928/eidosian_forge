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
def test_pooch_load_registry_comments():
    """Loading the registry from a file and strip line comments"""
    pup = Pooch(path='', base_url='')
    pup.load_registry(os.path.join(DATA_DIR, 'registry_comments.txt'))
    assert pup.registry == REGISTRY
    assert pup.registry_files.sort() == list(REGISTRY).sort()