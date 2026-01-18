import pathlib
from tempfile import gettempdir
import unittest
from traits.api import BaseDirectory, Directory, HasTraits, TraitError
def test_rejects_valid_pathlib_file(self):
    foo = ExistsBaseDirectory()
    with self.assertRaises(TraitError):
        foo.path = pathlib.Path(__file__)