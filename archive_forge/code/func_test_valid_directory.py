import pathlib
from tempfile import gettempdir
import unittest
from traits.api import BaseDirectory, Directory, HasTraits, TraitError
def test_valid_directory(self):
    example_model = ExampleModel(path=gettempdir())
    example_model.path = '.'