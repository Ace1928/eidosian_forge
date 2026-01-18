import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
def test_unsupported_protocol():
    """Should raise ValueError when protocol is not supported"""
    with pytest.raises(ValueError):
        choose_downloader('httpup://some-invalid-url.com')
    with pytest.raises(ValueError):
        choose_downloader('doii:XXX/XXX/file')