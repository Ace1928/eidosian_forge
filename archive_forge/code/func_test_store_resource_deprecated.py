import os
import tempfile
import unittest
from traits.util.resource import find_resource, store_resource
def test_store_resource_deprecated(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        with self.assertWarns(DeprecationWarning):
            store_resource('traits', os.path.join('traits', '__init__.py'), os.path.join(tmpdir, 'just_testing.py'))