import io
import pathlib
import unittest
import importlib_resources as resources
from . import data01
from . import util
def test_remove_in_context_manager(self):
    """
        It is not an error if the file that was temporarily stashed on the
        file system is removed inside the `with` stanza.
        """
    target = resources.files(self.data) / 'utf-8.file'
    with resources.as_file(target) as path:
        path.unlink()