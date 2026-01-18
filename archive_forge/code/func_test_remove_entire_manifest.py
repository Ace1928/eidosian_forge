import sys
import unittest
import os
from distutils.errors import DistutilsPlatformError
from distutils.tests import support
def test_remove_entire_manifest(self):
    from distutils.msvc9compiler import MSVCCompiler
    tempdir = self.mkdtemp()
    manifest = os.path.join(tempdir, 'manifest')
    f = open(manifest, 'w')
    try:
        f.write(_MANIFEST_WITH_ONLY_MSVC_REFERENCE)
    finally:
        f.close()
    compiler = MSVCCompiler()
    got = compiler._remove_visual_c_ref(manifest)
    self.assertIsNone(got)