import pathlib
from tempfile import gettempdir
import unittest
from traits.api import BaseDirectory, Directory, HasTraits, TraitError
class SimpleBaseDirectory(HasTraits):
    path = BaseDirectory(exists=False)