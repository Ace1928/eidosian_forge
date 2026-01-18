from ... import errors, tests, transport
from .. import index as _mod_index
def test_validate_empty(self):
    index = self.make_index()
    index.validate()