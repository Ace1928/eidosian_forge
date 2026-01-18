from ... import errors, tests, transport
from .. import index as _mod_index
def test_validate_reloads(self):
    idx, reload_counter = self.make_combined_index_with_missing()
    idx.validate()
    self.assertEqual([1, 1, 0], reload_counter)