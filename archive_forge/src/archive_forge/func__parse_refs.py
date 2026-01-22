import os
import tempfile
from io import BytesIO
from itertools import chain
from ...objects import hex_to_sha
from ...repo import Repo, check_ref_format
from .utils import CompatTestCase, require_git_version, rmtree_ro, run_git_or_fail
def _parse_refs(self, output):
    refs = {}
    for line in BytesIO(output):
        fields = line.rstrip(b'\n').split(b' ')
        self.assertEqual(3, len(fields))
        refname, type_name, sha = fields
        check_ref_format(refname[5:])
        hex_to_sha(sha)
        refs[refname] = (type_name, sha)
    return refs