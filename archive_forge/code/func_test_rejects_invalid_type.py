import pathlib
from tempfile import gettempdir
import unittest
from traits.api import BaseDirectory, Directory, HasTraits, TraitError
def test_rejects_invalid_type(self):
    """ Rejects instances that are not `str` or `os.PathLike`.
        """
    foo = ExistsBaseDirectory()
    with self.assertRaises(TraitError):
        foo.path = 1
    with self.assertRaises(TraitError):
        foo.path = b'!!!invalid_directory'