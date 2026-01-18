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
def test_pooch_download_retry_fails_eventually(monkeypatch):
    """Check that retrying the download fails after the set amount of retries"""
    with TemporaryDirectory() as local_store:
        monkeypatch.setattr(core, 'hash_matches', FakeHashMatches(3).hash_matches)
        path = Path(local_store)
        pup = Pooch(path=path, base_url=BASEURL, registry=REGISTRY, retry_if_failed=1)
        with pytest.raises(ValueError) as error:
            with capture_log() as log_file:
                pup.fetch('tiny-data.txt')
        logs = log_file.getvalue().strip().split('\n')
        assert len(logs) == 2
        assert logs[0].startswith('Downloading')
        assert logs[0].endswith(f"'{path}'.")
        assert 'Failed to download' in logs[1]
        assert 'download again 1 more time.' in logs[1]
        assert 'does not match the known hash' in str(error)