import sys
import unittest
import os
from distutils.errors import DistutilsPlatformError
from distutils.tests import support
@unittest.skipUnless(SKIP_MESSAGE is None, SKIP_MESSAGE)
class msvc9compilerTestCase(support.TempdirManager, unittest.TestCase):

    def test_no_compiler(self):
        from distutils.msvc9compiler import query_vcvarsall

        def _find_vcvarsall(version):
            return None
        from distutils import msvc9compiler
        old_find_vcvarsall = msvc9compiler.find_vcvarsall
        msvc9compiler.find_vcvarsall = _find_vcvarsall
        try:
            self.assertRaises(DistutilsPlatformError, query_vcvarsall, 'wont find this version')
        finally:
            msvc9compiler.find_vcvarsall = old_find_vcvarsall

    def test_reg_class(self):
        from distutils.msvc9compiler import Reg
        self.assertRaises(KeyError, Reg.get_value, 'xxx', 'xxx')
        path = 'Control Panel\\Desktop'
        v = Reg.get_value(path, 'dragfullwindows')
        self.assertIn(v, ('0', '1', '2'))
        import winreg
        HKCU = winreg.HKEY_CURRENT_USER
        keys = Reg.read_keys(HKCU, 'xxxx')
        self.assertEqual(keys, None)
        keys = Reg.read_keys(HKCU, 'Control Panel')
        self.assertIn('Desktop', keys)

    def test_remove_visual_c_ref(self):
        from distutils.msvc9compiler import MSVCCompiler
        tempdir = self.mkdtemp()
        manifest = os.path.join(tempdir, 'manifest')
        f = open(manifest, 'w')
        try:
            f.write(_MANIFEST_WITH_MULTIPLE_REFERENCES)
        finally:
            f.close()
        compiler = MSVCCompiler()
        compiler._remove_visual_c_ref(manifest)
        f = open(manifest)
        try:
            content = '\n'.join([line.rstrip() for line in f.readlines()])
        finally:
            f.close()
        self.assertEqual(content, _CLEANED_MANIFEST)

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