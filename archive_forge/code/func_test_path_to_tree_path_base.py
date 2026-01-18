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
def test_path_to_tree_path_base(self):
    self.assertEqual(b'bar', porcelain.path_to_tree_path(self.test_dir, self.fp))
    self.assertEqual(b'bar', porcelain.path_to_tree_path('.', './bar'))
    self.assertEqual(b'bar', porcelain.path_to_tree_path('.', 'bar'))
    cwd = os.getcwd()
    self.assertEqual(b'bar', porcelain.path_to_tree_path('.', os.path.join(cwd, 'bar')))
    self.assertEqual(b'bar', porcelain.path_to_tree_path(cwd, 'bar'))