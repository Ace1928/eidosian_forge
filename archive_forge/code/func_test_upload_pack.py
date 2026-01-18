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
def test_upload_pack(self):
    outf = BytesIO()
    exitcode = porcelain.upload_pack(self.repo.path, BytesIO(b'0000'), outf)
    outlines = outf.getvalue().splitlines()
    self.assertEqual([b'0000'], outlines)
    self.assertEqual(0, exitcode)