from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_repr_with_name(self):
    """
        If the fake has a C{name} property it's included in the repr string to
        make it easier to figure out what it is.
        """
    self.launchpad.me = dict(name='foo')
    person = self.launchpad.me
    self.assertEqual('<FakeEntry person foo at %s>' % hex(id(person)), repr(person))