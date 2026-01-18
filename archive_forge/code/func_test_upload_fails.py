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
def test_upload_fails(self):
    self.next_msg = 'Not Found'
    self.next_code = 404
    self.assertRaises(DistutilsError, self.test_upload)