import copy
import http.server
import os
import select
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from contextlib import suppress
from io import BytesIO
from urllib.parse import unquote
from dulwich import client, file, index, objects, protocol, repo
from dulwich.tests import SkipTest, expectedFailure
from .utils import (
def test_send_new_branch_empty_pack(self):
    with repo.Repo(os.path.join(self.gitroot, 'dest')) as dest:
        dummy_commit = self.make_dummy_commit(dest)
        dest.refs[b'refs/heads/master'] = dummy_commit
        dest.refs[b'refs/heads/abranch'] = dummy_commit
        sendrefs = {b'refs/heads/bbranch': dummy_commit}

        def gen_pack(have, want, ofs_delta=False, progress=None):
            return (0, [])
        c = self._client()
        self.assertEqual(dest.refs[b'refs/heads/abranch'], dummy_commit)
        c.send_pack(self._build_path('/dest'), lambda _: sendrefs, gen_pack)
        self.assertEqual(dummy_commit, dest.refs[b'refs/heads/abranch'])