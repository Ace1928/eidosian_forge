import gyp.generator.xcode as xcode
import unittest
import sys
class TestEscapeXcodeDefine(unittest.TestCase):
    if sys.platform == 'darwin':

        def test_InheritedRemainsUnescaped(self):
            self.assertEqual(xcode.EscapeXcodeDefine('$(inherited)'), '$(inherited)')

        def test_Escaping(self):
            self.assertEqual(xcode.EscapeXcodeDefine('a b"c\\'), 'a\\ b\\"c\\\\')