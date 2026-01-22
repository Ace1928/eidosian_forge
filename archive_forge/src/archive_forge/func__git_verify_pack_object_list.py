import binascii
import os
import re
import shutil
import tempfile
from dulwich.tests import SkipTest
from ...objects import Blob
from ...pack import write_pack
from ..test_pack import PackTests, a_sha, pack1_sha
from .utils import require_git_version, run_git_or_fail
def _git_verify_pack_object_list(output):
    pack_shas = set()
    for line in output.splitlines():
        sha = line[:40]
        try:
            binascii.unhexlify(sha)
        except (TypeError, binascii.Error):
            continue
        pack_shas.add(sha)
    return pack_shas