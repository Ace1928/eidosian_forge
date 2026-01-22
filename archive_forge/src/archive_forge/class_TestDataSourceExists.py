import os
import pytest
from tempfile import mkdtemp, mkstemp, NamedTemporaryFile
from shutil import rmtree
import numpy.lib._datasource as datasource
from numpy.testing import assert_, assert_equal, assert_raises
import urllib.request as urllib_request
from urllib.parse import urlparse
from urllib.error import URLError
class TestDataSourceExists:

    def setup_method(self):
        self.tmpdir = mkdtemp()
        self.ds = datasource.DataSource(self.tmpdir)

    def teardown_method(self):
        rmtree(self.tmpdir)
        del self.ds

    def test_ValidHTTP(self):
        assert_(self.ds.exists(valid_httpurl()))

    def test_InvalidHTTP(self):
        assert_equal(self.ds.exists(invalid_httpurl()), False)

    def test_ValidFile(self):
        tmpfile = valid_textfile(self.tmpdir)
        assert_(self.ds.exists(tmpfile))
        localdir = mkdtemp()
        tmpfile = valid_textfile(localdir)
        assert_(self.ds.exists(tmpfile))
        rmtree(localdir)

    def test_InvalidFile(self):
        tmpfile = invalid_textfile(self.tmpdir)
        assert_equal(self.ds.exists(tmpfile), False)