import pathlib
from tempfile import gettempdir
import unittest
from traits.api import BaseDirectory, Directory, HasTraits, TraitError
def test_rejects_valid_file_name(self):
    foo = ExistsBaseDirectory()
    with self.assertRaises(TraitError):
        foo.path = __file__