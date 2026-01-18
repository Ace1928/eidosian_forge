import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
def test_put_object_fails(self):
    with patch('geventhttpclient.HTTPClient.request', lambda *args, **kwargs: Response(status=400)):
        self.assertRaises(swift.SwiftException, lambda: self.conn.put_object('a', BytesIO(b'content')))