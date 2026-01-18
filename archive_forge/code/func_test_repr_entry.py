from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_repr_entry(self):
    """A custom C{__repr__} is provided for L{FakeEntry}s."""
    bug = dict()
    self.launchpad.bugs = dict(entries=[bug])
    [bug] = list(self.launchpad.bugs)
    self.assertEqual('<FakeEntry bug object at %s>' % hex(id(bug)), repr(bug))