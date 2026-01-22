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
class LsFilesTests(PorcelainTestCase):

    def test_empty(self):
        self.assertEqual([], list(porcelain.ls_files(self.repo)))

    def test_simple(self):
        fullpath = os.path.join(self.repo.path, 'foo')
        with open(fullpath, 'w') as f:
            f.write('origstuff')
        porcelain.add(repo=self.repo.path, paths=[fullpath])
        self.assertEqual([b'foo'], list(porcelain.ls_files(self.repo)))