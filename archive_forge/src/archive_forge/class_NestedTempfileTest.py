import os
import tempfile
import testtools
from testtools.matchers import StartsWith
from fixtures import (
class NestedTempfileTest(testtools.TestCase):
    """Tests for `NestedTempfile`."""

    def test_normal(self):
        starting_tempdir = tempfile.gettempdir()
        with NestedTempfile():
            self.assertEqual(tempfile.tempdir, tempfile.gettempdir())
            self.assertNotEqual(starting_tempdir, tempfile.tempdir)
            self.assertTrue(os.path.isdir(tempfile.tempdir))
            nested_tempdir = tempfile.tempdir
        self.assertEqual(tempfile.tempdir, tempfile.gettempdir())
        self.assertEqual(starting_tempdir, tempfile.tempdir)
        self.assertFalse(os.path.isdir(nested_tempdir))

    def test_exception(self):

        class ContrivedException(Exception):
            pass
        try:
            with NestedTempfile():
                nested_tempdir = tempfile.tempdir
                raise ContrivedException
        except ContrivedException:
            self.assertFalse(os.path.isdir(nested_tempdir))