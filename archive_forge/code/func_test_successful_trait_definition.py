import contextlib
import io
import os
import shutil
import tempfile
import textwrap
import tokenize
import unittest
import unittest.mock as mock
from traits.api import Bool, HasTraits, Int, Property
from traits.testing.optional_dependencies import sphinx, requires_sphinx
def test_successful_trait_definition(self):
    definition = trait_definition(cls=Fake, trait_name='test_attribute')
    self.assertEqual(definition, 'Property(Bool, label="ミスあり")')