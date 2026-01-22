import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _CaseInsensitiveFilesystemFeature(Feature):
    """Check if underlying filesystem is case-insensitive but *not* case
    preserving.
    """

    def _probe(self):
        if CaseInsCasePresFilenameFeature.available():
            return False
        from breezy import tests
        if tests.TestCaseWithMemoryTransport.TEST_ROOT is None:
            root = tempfile.mkdtemp(prefix='testbzr-', suffix='.tmp')
            tests.TestCaseWithMemoryTransport.TEST_ROOT = root
        else:
            root = tests.TestCaseWithMemoryTransport.TEST_ROOT
        tdir = tempfile.mkdtemp(prefix='case-sensitive-probe-', suffix='', dir=root)
        name_a = osutils.pathjoin(tdir, 'a')
        name_A = osutils.pathjoin(tdir, 'A')
        os.mkdir(name_a)
        result = osutils.isdir(name_A)
        tests._rmtree_temp_dir(tdir)
        return result

    def feature_name(self):
        return 'case-insensitive filesystem'