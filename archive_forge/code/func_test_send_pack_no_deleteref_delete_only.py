import base64
import os
import shutil
import sys
import tempfile
import warnings
from io import BytesIO
from typing import Dict
from unittest.mock import patch
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
import dulwich
from dulwich import client
from dulwich.tests import TestCase, skipIf
from ..client import (
from ..config import ConfigDict
from ..objects import Commit, Tree
from ..pack import pack_objects_to_data, write_pack_data, write_pack_objects
from ..protocol import TCP_GIT_PORT, Protocol
from ..repo import MemoryRepo, Repo
from .utils import open_repo, setup_warning_catcher, tear_down_repo
def test_send_pack_no_deleteref_delete_only(self):
    pkts = [b'310ca9477129b8586fa2afc779c1f57cf64bba6c refs/heads/master\x00 report-status ofs-delta\n', b'', b'']
    for pkt in pkts:
        if pkt == b'':
            self.rin.write(b'0000')
        else:
            self.rin.write(('%04x' % (len(pkt) + 4)).encode('ascii') + pkt)
    self.rin.seek(0)

    def update_refs(refs):
        return {b'refs/heads/master': b'0' * 40}

    def generate_pack_data(have, want, ofs_delta=False, progress=None):
        return (0, [])
    result = self.client.send_pack(b'/', update_refs, generate_pack_data)
    self.assertEqual(result.ref_status, {b'refs/heads/master': 'remote does not support deleting refs'})
    self.assertEqual(result.refs, {b'refs/heads/master': b'310ca9477129b8586fa2afc779c1f57cf64bba6c'})
    self.assertEqual(self.rout.getvalue(), b'0000')