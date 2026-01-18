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
@pytest.mark.parametrize('url', [BASEURL, FIGSHAREURL, ZENODOURL, DATAVERSEURL], ids=['https', 'figshare', 'zenodo', 'dataverse'])
def test_pooch_custom_url(url):
    """Have pooch download the file from URL that is not base_url"""
    with TemporaryDirectory() as local_store:
        path = Path(local_store)
        urls = {'tiny-data.txt': url + 'tiny-data.txt'}
        pup = Pooch(path=path, base_url='', registry=REGISTRY, urls=urls)
        with capture_log() as log_file:
            fname = pup.fetch('tiny-data.txt')
            logs = log_file.getvalue()
            assert logs.split()[0] == 'Downloading'
            assert logs.split()[-1] == f"'{path}'."
        check_tiny_data(fname)
        with capture_log() as log_file:
            fname = pup.fetch('tiny-data.txt')
            assert log_file.getvalue() == ''