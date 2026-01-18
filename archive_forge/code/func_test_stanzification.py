import os
from ... import tests
from ...conflicts import resolve
from ...tests import scenarios
from ...tests.test_conflicts import vary_by_conflicts
from .. import conflicts as bzr_conflicts
def test_stanzification(self):
    stanza = self.conflict.as_stanza()
    if 'file_id' in stanza:
        self.assertStartsWith(stanza['file_id'], 'îd')
    self.assertStartsWith(stanza['path'], 'påth')
    if 'conflict_path' in stanza:
        self.assertStartsWith(stanza['conflict_path'], 'påth')
    if 'conflict_file_id' in stanza:
        self.assertStartsWith(stanza['conflict_file_id'], 'îd')