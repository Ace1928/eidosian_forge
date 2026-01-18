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
@pytest.mark.network
def test_retrieve_fname():
    """Try downloading some data with retrieve and setting the file name"""
    with TemporaryDirectory() as local_store:
        data_file = 'tiny-data.txt'
        url = BASEURL + data_file
        with capture_log() as log_file:
            fname = retrieve(url, known_hash=None, path=local_store, fname=data_file)
            logs = log_file.getvalue()
            assert logs.split()[0] == 'Downloading'
            assert 'SHA256 hash of downloaded file:' in logs
            assert REGISTRY[data_file] in logs
        assert data_file == os.path.split(fname)[1]
        check_tiny_data(fname)
        assert file_hash(fname) == REGISTRY[data_file]