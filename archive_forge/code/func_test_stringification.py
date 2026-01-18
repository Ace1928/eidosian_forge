import os
from ... import tests
from ...conflicts import resolve
from ...tests import scenarios
from ...tests.test_conflicts import vary_by_conflicts
from .. import conflicts as bzr_conflicts
def test_stringification(self):
    text = str(self.conflict)
    self.assertContainsString(text, self.conflict.path)
    self.assertContainsString(text.lower(), 'conflict')
    self.assertContainsString(repr(self.conflict), self.conflict.__class__.__name__)