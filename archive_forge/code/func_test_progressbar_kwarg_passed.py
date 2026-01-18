import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
@pytest.mark.skipif(tqdm is None, reason='requires tqdm')
@pytest.mark.parametrize('url', [BASEURL + 'tiny-data.txt', FIGSHAREURL])
def test_progressbar_kwarg_passed(url):
    """The progressbar keyword argument must pass through choose_downloader"""
    downloader = choose_downloader(url, progressbar=True)
    assert downloader.progressbar is True