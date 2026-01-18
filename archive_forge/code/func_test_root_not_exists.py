import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
def test_root_not_exists(self):
    with patch('geventhttpclient.HTTPClient.request', lambda *args: Response(status=404)):
        self.assertEqual(self.conn.test_root_exists(), None)