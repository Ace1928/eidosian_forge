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
class FakeOpen(object):

    def __init__(self, url, msg=None, code=None):
        self.url = url
        if not isinstance(url, str):
            self.req = url
        else:
            self.req = None
        self.msg = msg or 'OK'
        self.code = code or 200

    def getheader(self, name, default=None):
        return {'content-type': 'text/plain; charset=utf-8'}.get(name.lower(), default)

    def read(self):
        return b'xyzzy'

    def getcode(self):
        return self.code