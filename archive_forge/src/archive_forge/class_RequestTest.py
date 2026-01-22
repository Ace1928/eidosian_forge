import base64
import pytest
import struct
from winrm.encryption import Encryption
from winrm.exceptions import WinRMError
class RequestTest(object):

    def __init__(self):
        self.url = 'http://testhost.com/path'