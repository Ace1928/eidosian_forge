import os
import subprocess
import sys
from io import BytesIO
from dulwich.repo import Repo
from ...tests import TestCaseWithTransport
from ...tests.features import PathFeature
from ..git_remote_helper import RemoteHelper, fetch, open_local_dir
from ..object_store import get_object_store
from . import FastimportFeature
class RemoteHelperTests(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.local_dir = self.make_branch_and_tree('local', format='git').controldir
        self.remote_tree = self.make_branch_and_tree('remote')
        self.remote_dir = self.remote_tree.controldir
        self.shortname = 'bzr'
        self.helper = RemoteHelper(self.local_dir, self.shortname, self.remote_dir)

    def test_capabilities(self):
        f = BytesIO()
        self.helper.cmd_capabilities(f, [])
        capabs = f.getvalue()
        base = b'fetch\noption\npush\n'
        self.assertTrue(capabs in (base + b'\n', base + b'import\nrefspec *:*\n\n'), capabs)

    def test_option(self):
        f = BytesIO()
        self.helper.cmd_option(f, [])
        self.assertEqual(b'unsupported\n', f.getvalue())

    def test_list_basic(self):
        f = BytesIO()
        self.helper.cmd_list(f, [])
        self.assertEqual(b'\n', f.getvalue())

    def test_import(self):
        self.requireFeature(FastimportFeature)
        self.build_tree_contents([('remote/afile', b'somecontent')])
        self.remote_tree.add(['afile'])
        self.remote_tree.commit(b'A commit message', timestamp=1330445983, timezone=0, committer=b'Somebody <jrandom@example.com>')
        f = BytesIO()
        self.helper.cmd_import(f, ['import', 'refs/heads/master'])
        self.assertEqual(b'reset refs/heads/master\ncommit refs/heads/master\nmark :1\ncommitter Somebody <jrandom@example.com> 1330445983 +0000\ndata 16\nA commit message\nM 644 inline afile\ndata 11\nsomecontent\n', f.getvalue())