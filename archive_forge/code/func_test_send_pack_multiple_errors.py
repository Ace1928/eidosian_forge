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
def test_send_pack_multiple_errors(self):
    dest, dummy = self.disable_ff_and_make_dummy_commit()
    branch, master = (b'refs/heads/branch', b'refs/heads/master')
    dest.refs[branch] = dest.refs[master] = dummy
    repo_dir = os.path.join(self.gitroot, 'server_new.export')
    with repo.Repo(repo_dir) as src:
        sendrefs, gen_pack = self.compute_send(src)
        c = self._client()
        result = c.send_pack(self._build_path('/dest'), lambda _: sendrefs, gen_pack)
        self.assertEqual({branch: 'non-fast-forward', master: 'non-fast-forward'}, result.ref_status)