import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
Read data in blocks and verify that the reads are not larger than
           the maximum read size.
        