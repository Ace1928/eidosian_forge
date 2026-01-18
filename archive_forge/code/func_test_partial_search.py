from ... import graph as _mod_graph
from ... import tests
from ...revision import NULL_REVISION
from ...tests.test_graph import TestGraphBase
from .. import vf_search
def test_partial_search(self):
    parent_map = {k: extended_history_shortcut[k] for k in [b'e', b'f']}
    self.assertSearchResult([b'e', b'f'], [b'd', b'a'], 2, parent_map)
    parent_map.update(((k, extended_history_shortcut[k]) for k in [b'd', b'a']))
    self.assertSearchResult([b'e', b'f'], [b'c', NULL_REVISION], 4, parent_map)
    parent_map[b'c'] = extended_history_shortcut[b'c']
    self.assertSearchResult([b'e', b'f'], [b'b'], 6, parent_map, missing_keys=[NULL_REVISION])
    parent_map[b'b'] = extended_history_shortcut[b'b']
    self.assertSearchResult([b'e', b'f'], [], 7, parent_map, missing_keys=[NULL_REVISION])