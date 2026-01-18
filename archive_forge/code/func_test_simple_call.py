import os
import pickle
import pickletools
import shutil
import tempfile
import unittest
import unittest.mock as mock
import warnings
from traits.api import HasTraits, Int
from traits.testing.optional_dependencies import requires_traitsui, traitsui
def test_simple_call(self):
    model = Model()
    with mock.patch.object(self.toolkit, 'view_application') as mock_view:
        model.configure_traits()
    self.assertEqual(mock_view.call_count, 1)