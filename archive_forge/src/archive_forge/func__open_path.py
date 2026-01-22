import io
import json
import optparse
import os.path
import sys
from errno import EEXIST
from textwrap import dedent
from testtools import StreamToDict
from subunit.filters import run_tests_from_stream
def _open_path(root, subpath):
    name = _allocate_path(root, subpath)
    try:
        os.makedirs(os.path.dirname(name))
    except (OSError, IOError) as e:
        if e.errno != EEXIST:
            raise
    return io.open(name, 'wb')