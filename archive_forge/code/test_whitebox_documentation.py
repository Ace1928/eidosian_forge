import os
import tempfile
import breezy
from .. import errors, osutils, tests
from ..osutils import abspath, pathjoin, realpath, relpath
test for branch path lookups

        breezy.osutils._relpath do a simple but subtle
        job: given a path (either relative to cwd or absolute), work out
        if it is inside a branch and return the path relative to the base.
        