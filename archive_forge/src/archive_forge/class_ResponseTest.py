import base64
import pytest
import struct
from winrm.encryption import Encryption
from winrm.exceptions import WinRMError
class ResponseTest(object):

    def __init__(self, content_type, content):
        self.headers = {'Content-Type': content_type}
        self.content = content
        self.text = content
        self.request = RequestTest()