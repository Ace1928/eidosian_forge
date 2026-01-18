import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_readlines(self):
    f = response.ResponseFile('many', BytesIO(b'0\n1\nboo!\n'))
    self.assertEqual([b'0\n', b'1\n', b'boo!\n'], f.readlines())