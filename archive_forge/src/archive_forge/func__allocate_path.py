import io
import json
import optparse
import os.path
import sys
from errno import EEXIST
from textwrap import dedent
from testtools import StreamToDict
from subunit.filters import run_tests_from_stream
def _allocate_path(root, sub):
    """Figoure a path for sub under root.

    If sub tries to escape root, squash it with prejuidice.

    If the path already exists, a numeric suffix is appended.
    E.g. foo, foo-1, foo-2, etc.

    :return: the full path to sub.
    """
    candidate = os.path.realpath(os.path.join(root, sub))
    realroot = os.path.realpath(root)
    if not candidate.startswith(realroot):
        sub = sub.replace('/', '_').replace('\\', '_')
        return _allocate_path(root, sub)
    attempt = 0
    probe = candidate
    while os.path.exists(probe):
        attempt += 1
        probe = '%s-%s' % (candidate, attempt)
    return probe