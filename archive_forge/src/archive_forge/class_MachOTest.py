import os
import sys
import unittest
from ctypes.macholib.dyld import dyld_find
from ctypes.macholib.dylib import dylib_info
from ctypes.macholib.framework import framework_info
class MachOTest(unittest.TestCase):

    @unittest.skipUnless(sys.platform == 'darwin', 'OSX-specific test')
    def test_find(self):
        self.assertEqual(dyld_find('libSystem.dylib'), '/usr/lib/libSystem.dylib')
        self.assertEqual(dyld_find('System.framework/System'), '/System/Library/Frameworks/System.framework/System')
        self.assertIn(find_lib('pthread'), ('/usr/lib/libSystem.B.dylib', '/usr/lib/libpthread.dylib'))
        result = find_lib('z')
        self.assertRegex(result, '.*/lib/libz.*\\.dylib')
        self.assertIn(find_lib('IOKit'), ('/System/Library/Frameworks/IOKit.framework/Versions/A/IOKit', '/System/Library/Frameworks/IOKit.framework/IOKit'))

    @unittest.skipUnless(sys.platform == 'darwin', 'OSX-specific test')
    def test_info(self):
        self.assertIsNone(dylib_info('completely/invalid'))
        self.assertIsNone(dylib_info('completely/invalide_debug'))
        self.assertEqual(dylib_info('P/Foo.dylib'), d('P', 'Foo.dylib', 'Foo'))
        self.assertEqual(dylib_info('P/Foo_debug.dylib'), d('P', 'Foo_debug.dylib', 'Foo', suffix='debug'))
        self.assertEqual(dylib_info('P/Foo.A.dylib'), d('P', 'Foo.A.dylib', 'Foo', 'A'))
        self.assertEqual(dylib_info('P/Foo_debug.A.dylib'), d('P', 'Foo_debug.A.dylib', 'Foo_debug', 'A'))
        self.assertEqual(dylib_info('P/Foo.A_debug.dylib'), d('P', 'Foo.A_debug.dylib', 'Foo', 'A', 'debug'))

    @unittest.skipUnless(sys.platform == 'darwin', 'OSX-specific test')
    def test_framework_info(self):
        self.assertIsNone(framework_info('completely/invalid'))
        self.assertIsNone(framework_info('completely/invalid/_debug'))
        self.assertIsNone(framework_info('P/F.framework'))
        self.assertIsNone(framework_info('P/F.framework/_debug'))
        self.assertEqual(framework_info('P/F.framework/F'), d('P', 'F.framework/F', 'F'))
        self.assertEqual(framework_info('P/F.framework/F_debug'), d('P', 'F.framework/F_debug', 'F', suffix='debug'))
        self.assertIsNone(framework_info('P/F.framework/Versions'))
        self.assertIsNone(framework_info('P/F.framework/Versions/A'))
        self.assertEqual(framework_info('P/F.framework/Versions/A/F'), d('P', 'F.framework/Versions/A/F', 'F', 'A'))
        self.assertEqual(framework_info('P/F.framework/Versions/A/F_debug'), d('P', 'F.framework/Versions/A/F_debug', 'F', 'A', 'debug'))