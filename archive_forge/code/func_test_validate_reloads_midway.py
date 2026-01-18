from ... import errors, tests, transport
from .. import index as _mod_index
def test_validate_reloads_midway(self):
    idx, reload_counter = self.make_combined_index_with_missing(['2'])
    idx.validate()