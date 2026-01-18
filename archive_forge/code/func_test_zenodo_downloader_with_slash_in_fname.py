import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
@pytest.mark.network
def test_zenodo_downloader_with_slash_in_fname():
    """
    Test the Zenodo downloader when the path contains a forward slash

    Related to issue #336
    """
    with TemporaryDirectory() as local_store:
        base_url = ZENODOURL_W_SLASH + 'santisoler/pooch-test-data-v1.zip'
        downloader = DOIDownloader()
        outfile = os.path.join(local_store, 'test-data.zip')
        downloader(base_url, outfile, None)
        fnames = Unzip()(outfile, action='download', pooch=None)
        fname, = [f for f in fnames if 'tiny-data.txt' in f]
        check_tiny_data(fname)