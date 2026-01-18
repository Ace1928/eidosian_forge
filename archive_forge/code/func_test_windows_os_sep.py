import os
import pytest
from tempfile import mkdtemp, mkstemp, NamedTemporaryFile
from shutil import rmtree
import numpy.lib._datasource as datasource
from numpy.testing import assert_, assert_equal, assert_raises
import urllib.request as urllib_request
from urllib.parse import urlparse
from urllib.error import URLError
def test_windows_os_sep(self):
    orig_os_sep = os.sep
    try:
        os.sep = '\\'
        self.test_ValidHTTP()
        self.test_sandboxing()
    finally:
        os.sep = orig_os_sep