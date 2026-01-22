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
class TagListTests(PorcelainTestCase):

    def test_empty(self):
        tags = porcelain.tag_list(self.repo.path)
        self.assertEqual([], tags)

    def test_simple(self):
        self.repo.refs[b'refs/tags/foo'] = b'aa' * 20
        self.repo.refs[b'refs/tags/bar/bla'] = b'bb' * 20
        tags = porcelain.tag_list(self.repo.path)
        self.assertEqual([b'bar/bla', b'foo'], tags)