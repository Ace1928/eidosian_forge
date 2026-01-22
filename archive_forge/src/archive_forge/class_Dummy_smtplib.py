import sys
import io
import random
import mimetypes
import time
import os
import shutil
import smtplib
import shlex
import re
import subprocess
from urllib.parse import urlencode
from urllib import parse as urlparse
from http.cookies import BaseCookie
from paste import wsgilib
from paste import lint
from paste.response import HeaderDict
class Dummy_smtplib(object):
    existing = None

    def __init__(self, server):
        import warnings
        warnings.warn('Dummy_smtplib is not maintained and is deprecated', DeprecationWarning, 2)
        assert not self.existing, 'smtplib.SMTP() called again before Dummy_smtplib.existing.reset() called.'
        self.server = server
        self.open = True
        self.__class__.existing = self

    def quit(self):
        assert self.open, 'Called %s.quit() twice' % self
        self.open = False

    def sendmail(self, from_address, to_addresses, msg):
        self.from_address = from_address
        self.to_addresses = to_addresses
        self.message = msg

    def install(cls):
        smtplib.SMTP = cls
    install = classmethod(install)

    def reset(self):
        assert not self.open, 'SMTP connection not quit'
        self.__class__.existing = None