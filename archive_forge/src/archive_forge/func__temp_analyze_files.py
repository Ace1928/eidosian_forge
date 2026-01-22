import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
@pytest.fixture()
def _temp_analyze_files(tmpdir):
    """Generate temporary analyze file pair."""
    orig_img = tmpdir.join('orig.img')
    orig_hdr = tmpdir.join('orig.hdr')
    orig_img.open('w+').close()
    orig_hdr.open('w+').close()
    return (str(orig_img), str(orig_hdr))