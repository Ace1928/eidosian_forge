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
def test_some(self):
    cid = porcelain.commit(repo=self.repo.path, message=b'test status', author=b'author <email>', committer=b'committer <email>')
    self.assertEqual({b'refs/heads/master': cid, b'HEAD': cid}, porcelain.ls_remote(self.repo.path))