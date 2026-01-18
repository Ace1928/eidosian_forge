from unittest import TestCase
import textwrap
from jsonschema import exceptions
from jsonschema.validators import _LATEST_VERSION
def test_one_item_paths(self):
    self.assertShows("\n            Failed validating 'type' in schema:\n                {'type': 'string'}\n\n            On instance[0]:\n                5\n            ", path=[0], schema_path=['items'])