import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
class ShaFileCopyTests(TestCase):

    def assert_copy(self, orig):
        oclass = object_class(orig.type_num)
        copy = orig.copy()
        self.assertIsInstance(copy, oclass)
        self.assertEqual(copy, orig)
        self.assertIsNot(copy, orig)

    def test_commit_copy(self):
        attrs = {'tree': b'd80c186a03f423a81b39df39dc87fd269736ca86', 'parents': [b'ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd', b'4cffe90e0a41ad3f5190079d7c8f036bde29cbe6'], 'author': b'James Westby <jw+debian@jameswestby.net>', 'committer': b'James Westby <jw+debian@jameswestby.net>', 'commit_time': 1174773719, 'author_time': 1174773719, 'commit_timezone': 0, 'author_timezone': 0, 'message': b'Merge ../b\n'}
        commit = make_commit(**attrs)
        self.assert_copy(commit)

    def test_blob_copy(self):
        blob = make_object(Blob, data=b'i am a blob')
        self.assert_copy(blob)

    def test_tree_copy(self):
        blob = make_object(Blob, data=b'i am a blob')
        tree = Tree()
        tree[b'blob'] = (stat.S_IFREG, blob.id)
        self.assert_copy(tree)

    def test_tag_copy(self):
        tag = make_object(Tag, name=b'tag', message=b'', tagger=b'Tagger <test@example.com>', tag_time=12345, tag_timezone=0, object=(Commit, b'0' * 40))
        self.assert_copy(tag)