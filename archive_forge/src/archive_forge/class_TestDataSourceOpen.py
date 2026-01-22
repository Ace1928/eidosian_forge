import os
import pytest
from tempfile import mkdtemp, mkstemp, NamedTemporaryFile
from shutil import rmtree
import numpy.lib._datasource as datasource
from numpy.testing import assert_, assert_equal, assert_raises
import urllib.request as urllib_request
from urllib.parse import urlparse
from urllib.error import URLError
class TestDataSourceOpen:

    def setup_method(self):
        self.tmpdir = mkdtemp()
        self.ds = datasource.DataSource(self.tmpdir)

    def teardown_method(self):
        rmtree(self.tmpdir)
        del self.ds

    def test_ValidHTTP(self):
        fh = self.ds.open(valid_httpurl())
        assert_(fh)
        fh.close()

    def test_InvalidHTTP(self):
        url = invalid_httpurl()
        assert_raises(OSError, self.ds.open, url)
        try:
            self.ds.open(url)
        except OSError as e:
            assert_(e.errno is None)

    def test_InvalidHTTPCacheURLError(self):
        assert_raises(URLError, self.ds._cache, invalid_httpurl())

    def test_ValidFile(self):
        local_file = valid_textfile(self.tmpdir)
        fh = self.ds.open(local_file)
        assert_(fh)
        fh.close()

    def test_InvalidFile(self):
        invalid_file = invalid_textfile(self.tmpdir)
        assert_raises(OSError, self.ds.open, invalid_file)

    def test_ValidGzipFile(self):
        try:
            import gzip
        except ImportError:
            pytest.skip()
        filepath = os.path.join(self.tmpdir, 'foobar.txt.gz')
        fp = gzip.open(filepath, 'w')
        fp.write(magic_line)
        fp.close()
        fp = self.ds.open(filepath)
        result = fp.readline()
        fp.close()
        assert_equal(magic_line, result)

    def test_ValidBz2File(self):
        try:
            import bz2
        except ImportError:
            pytest.skip()
        filepath = os.path.join(self.tmpdir, 'foobar.txt.bz2')
        fp = bz2.BZ2File(filepath, 'w')
        fp.write(magic_line)
        fp.close()
        fp = self.ds.open(filepath)
        result = fp.readline()
        fp.close()
        assert_equal(magic_line, result)