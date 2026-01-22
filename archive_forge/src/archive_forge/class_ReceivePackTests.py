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
class ReceivePackTests(PorcelainTestCase):
    """Tests for receive_pack."""

    def test_receive_pack(self):
        filename = 'foo'
        fullpath = os.path.join(self.repo.path, filename)
        with open(fullpath, 'w') as f:
            f.write('stuff')
        porcelain.add(repo=self.repo.path, paths=fullpath)
        self.repo.do_commit(message=b'test status', author=b'author <email>', committer=b'committer <email>', author_timestamp=1402354300, commit_timestamp=1402354300, author_timezone=0, commit_timezone=0)
        outf = BytesIO()
        exitcode = porcelain.receive_pack(self.repo.path, BytesIO(b'0000'), outf)
        outlines = outf.getvalue().splitlines()
        self.assertEqual([b'0091319b56ce3aee2d489f759736a79cc552c9bb86d9 HEAD\x00 report-status delete-refs quiet ofs-delta side-band-64k no-done symref=HEAD:refs/heads/master', b'003f319b56ce3aee2d489f759736a79cc552c9bb86d9 refs/heads/master', b'0000'], outlines)
        self.assertEqual(0, exitcode)