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
class RemoteRemoveTests(PorcelainTestCase):

    def test_remove(self):
        porcelain.remote_add(self.repo, 'jelmer', 'git://jelmer.uk/code/dulwich')
        c = self.repo.get_config()
        self.assertEqual(c.get((b'remote', b'jelmer'), b'url'), b'git://jelmer.uk/code/dulwich')
        porcelain.remote_remove(self.repo, 'jelmer')
        self.assertRaises(KeyError, porcelain.remote_remove, self.repo, 'jelmer')
        c = self.repo.get_config()
        self.assertRaises(KeyError, c.get, (b'remote', b'jelmer'), b'url')