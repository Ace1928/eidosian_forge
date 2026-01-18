from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_replace_property(self):
    """Values already set on fake resource objects can be replaced."""
    self.launchpad.me = dict(name='foo')
    person = self.launchpad.me
    self.assertEqual('foo', person.name)
    person.name = 'bar'
    self.assertEqual('bar', person.name)
    self.assertEqual('bar', self.launchpad.me.name)