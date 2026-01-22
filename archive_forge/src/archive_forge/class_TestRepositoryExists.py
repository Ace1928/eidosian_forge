import os
import pytest
from tempfile import mkdtemp, mkstemp, NamedTemporaryFile
from shutil import rmtree
import numpy.lib._datasource as datasource
from numpy.testing import assert_, assert_equal, assert_raises
import urllib.request as urllib_request
from urllib.parse import urlparse
from urllib.error import URLError
class TestRepositoryExists:

    def setup_method(self):
        self.tmpdir = mkdtemp()
        self.repos = datasource.Repository(valid_baseurl(), self.tmpdir)

    def teardown_method(self):
        rmtree(self.tmpdir)
        del self.repos

    def test_ValidFile(self):
        tmpfile = valid_textfile(self.tmpdir)
        assert_(self.repos.exists(tmpfile))

    def test_InvalidFile(self):
        tmpfile = invalid_textfile(self.tmpdir)
        assert_equal(self.repos.exists(tmpfile), False)

    def test_RemoveHTTPFile(self):
        assert_(self.repos.exists(valid_httpurl()))

    def test_CachedHTTPFile(self):
        localfile = valid_httpurl()
        scheme, netloc, upath, pms, qry, frg = urlparse(localfile)
        local_path = os.path.join(self.repos._destpath, netloc)
        os.mkdir(local_path, 448)
        tmpfile = valid_textfile(local_path)
        assert_(self.repos.exists(tmpfile))