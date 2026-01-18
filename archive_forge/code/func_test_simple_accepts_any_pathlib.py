import pathlib
from tempfile import gettempdir
import unittest
from traits.api import BaseDirectory, Directory, HasTraits, TraitError
def test_simple_accepts_any_pathlib(self):
    """ BaseDirectory with no existence check accepts any pathlib path.
        """
    foo = SimpleBaseDirectory()
    foo.path = pathlib.Path('!!!')
    self.assertIsInstance(foo.path, str)