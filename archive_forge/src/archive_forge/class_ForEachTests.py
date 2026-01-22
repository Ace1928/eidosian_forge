import contextlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from io import BytesIO, StringIO
from unittest import skipIf
from dulwich import porcelain
from dulwich.tests import TestCase
from ..diff_tree import tree_changes
from ..errors import CommitError
from ..objects import ZERO_SHA, Blob, Tag, Tree
from ..porcelain import CheckoutError
from ..repo import NoIndexPresent, Repo
from ..server import DictBackend
from ..web import make_server, make_wsgi_chain
from .utils import build_commit_graph, make_commit, make_object
class ForEachTests(PorcelainTestCase):

    def setUp(self):
        super().setUp()
        c1, c2, c3, c4 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2], [4]])
        porcelain.tag_create(self.repo.path, b'v0.1', objectish=c1.id, annotated=True, message=b'0.1')
        porcelain.tag_create(self.repo.path, b'v1.0', objectish=c2.id, annotated=True, message=b'1.0')
        porcelain.tag_create(self.repo.path, b'simple-tag', objectish=c3.id)
        porcelain.tag_create(self.repo.path, b'v1.1', objectish=c4.id, annotated=True, message=b'1.1')
        porcelain.branch_create(self.repo.path, b'feat', objectish=c2.id.decode('ascii'))
        self.repo.refs[b'HEAD'] = c4.id

    def test_for_each_ref(self):
        refs = porcelain.for_each_ref(self.repo)
        self.assertEqual([(object_type, tag) for _, object_type, tag in refs], [(b'commit', b'refs/heads/feat'), (b'commit', b'refs/heads/master'), (b'commit', b'refs/tags/simple-tag'), (b'tag', b'refs/tags/v0.1'), (b'tag', b'refs/tags/v1.0'), (b'tag', b'refs/tags/v1.1')])

    def test_for_each_ref_pattern(self):
        versions = porcelain.for_each_ref(self.repo, pattern='refs/tags/v*')
        self.assertEqual([(object_type, tag) for _, object_type, tag in versions], [(b'tag', b'refs/tags/v0.1'), (b'tag', b'refs/tags/v1.0'), (b'tag', b'refs/tags/v1.1')])
        versions = porcelain.for_each_ref(self.repo, pattern='refs/tags/v1.?')
        self.assertEqual([(object_type, tag) for _, object_type, tag in versions], [(b'tag', b'refs/tags/v1.0'), (b'tag', b'refs/tags/v1.1')])