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
class FsckTests(PorcelainTestCase):

    def test_none(self):
        self.assertEqual([], list(porcelain.fsck(self.repo)))

    def test_git_dir(self):
        obj = Tree()
        a = Blob()
        a.data = b'foo'
        obj.add(b'.git', 33188, a.id)
        self.repo.object_store.add_objects([(a, None), (obj, None)])
        self.assertEqual([(obj.id, 'invalid name .git')], [(sha, str(e)) for sha, e in porcelain.fsck(self.repo)])