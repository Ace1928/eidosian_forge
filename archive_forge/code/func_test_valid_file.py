import os
from pathlib import Path
import unittest
from traits.api import File, HasTraits, TraitError
from traits.testing.optional_dependencies import requires_traitsui
def test_valid_file(self):
    example_model = ExampleModel(file_name=__file__)
    example_model.file_name = os.path.__file__