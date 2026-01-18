import os
import pytest
from tempfile import mkdtemp, mkstemp, NamedTemporaryFile
from shutil import rmtree
import numpy.lib._datasource as datasource
from numpy.testing import assert_, assert_equal, assert_raises
import urllib.request as urllib_request
from urllib.parse import urlparse
from urllib.error import URLError
def valid_textfile(filedir):
    fd, path = mkstemp(suffix='.txt', prefix='dstmp_', dir=filedir, text=True)
    os.close(fd)
    return path