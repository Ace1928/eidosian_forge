import unittest
class TestInterfaceImport(unittest.TestCase):

    def test_import(self):
        import zope.interface.common.interfaces as x
        self.assertIsNotNone(x)