import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
def test_put_named_file(self):
    store = {'fakerepo/objects/pack': ''}
    with patch('dulwich.contrib.swift.SwiftConnector', new_callable=create_swift_connector, store=store):
        repo = swift.SwiftRepo('fakerepo', conf=self.conf)
        desc = b'Fake repo'
        repo._put_named_file('description', desc)
    self.assertEqual(repo.scon.store['fakerepo/description'], desc)