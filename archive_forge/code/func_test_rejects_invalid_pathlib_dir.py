import pathlib
from tempfile import gettempdir
import unittest
from traits.api import BaseDirectory, Directory, HasTraits, TraitError
def test_rejects_invalid_pathlib_dir(self):
    foo = ExistsBaseDirectory()
    with self.assertRaises(TraitError):
        foo.path = pathlib.Path('!!!invalid_directory')