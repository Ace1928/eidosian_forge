import os
import unittest
import unittest.mock as mock
from urllib.error import HTTPError
from distutils.command import upload as upload_mod
from distutils.command.upload import upload
from distutils.core import Distribution
from distutils.errors import DistutilsError
from distutils.log import ERROR, INFO
from distutils.tests.test_config import PYPIRC, BasePyPIRCCommandTestCase
def test_upload_correct_cr(self):
    tmp = self.mkdtemp()
    path = os.path.join(tmp, 'xxx')
    self.write_file(path, content='yy\r')
    command, pyversion, filename = ('xxx', '2.6', path)
    dist_files = [(command, pyversion, filename)]
    self.write_file(self.rc, PYPIRC_LONG_PASSWORD)
    pkg_dir, dist = self.create_dist(dist_files=dist_files, description='long description\r')
    cmd = upload(dist)
    cmd.show_response = 1
    cmd.ensure_finalized()
    cmd.run()
    headers = dict(self.last_open.req.headers)
    self.assertGreaterEqual(int(headers['Content-length']), 2172)
    self.assertIn(b'long description\r', self.last_open.req.data)