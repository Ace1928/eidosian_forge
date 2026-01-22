import os
import pytest
from tempfile import mkdtemp, mkstemp, NamedTemporaryFile
from shutil import rmtree
import numpy.lib._datasource as datasource
from numpy.testing import assert_, assert_equal, assert_raises
import urllib.request as urllib_request
from urllib.parse import urlparse
from urllib.error import URLError
class TestRepositoryAbspath:

    def setup_method(self):
        self.tmpdir = os.path.abspath(mkdtemp())
        self.repos = datasource.Repository(valid_baseurl(), self.tmpdir)

    def teardown_method(self):
        rmtree(self.tmpdir)
        del self.repos

    def test_ValidHTTP(self):
        scheme, netloc, upath, pms, qry, frg = urlparse(valid_httpurl())
        local_path = os.path.join(self.repos._destpath, netloc, upath.strip(os.sep).strip('/'))
        filepath = self.repos.abspath(valid_httpfile())
        assert_equal(local_path, filepath)

    def test_sandboxing(self):
        tmp_path = lambda x: os.path.abspath(self.repos.abspath(x))
        assert_(tmp_path(valid_httpfile()).startswith(self.tmpdir))
        for fn in malicious_files:
            assert_(tmp_path(http_path + fn).startswith(self.tmpdir))
            assert_(tmp_path(fn).startswith(self.tmpdir))

    def test_windows_os_sep(self):
        orig_os_sep = os.sep
        try:
            os.sep = '\\'
            self.test_ValidHTTP()
            self.test_sandboxing()
        finally:
            os.sep = orig_os_sep