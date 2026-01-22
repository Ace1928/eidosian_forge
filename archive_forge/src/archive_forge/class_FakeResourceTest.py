from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
class FakeResourceTest(ResourcedTestCase):
    resources = [('launchpad', FakeLaunchpadResource())]

    def test_repr_entry(self):
        """A custom C{__repr__} is provided for L{FakeEntry}s."""
        bug = dict()
        self.launchpad.bugs = dict(entries=[bug])
        [bug] = list(self.launchpad.bugs)
        self.assertEqual('<FakeEntry bug object at %s>' % hex(id(bug)), repr(bug))

    def test_repr_collection(self):
        """A custom C{__repr__} is provided for L{FakeCollection}s."""
        branches = dict(total_size='test-branch')
        self.launchpad.me = dict(getBranches=lambda statuses: branches)
        branches = self.launchpad.me.getBranches([])
        obj_id = hex(id(branches))
        self.assertEqual('<FakeCollection branch-page-resource object at %s>' % obj_id, repr(branches))

    def test_repr_with_name(self):
        """
        If the fake has a C{name} property it's included in the repr string to
        make it easier to figure out what it is.
        """
        self.launchpad.me = dict(name='foo')
        person = self.launchpad.me
        self.assertEqual('<FakeEntry person foo at %s>' % hex(id(person)), repr(person))

    def test_repr_with_id(self):
        """
        If the fake has an C{id} property it's included in the repr string to
        make it easier to figure out what it is.
        """
        bug = dict(id='1', title='Bug #1')
        self.launchpad.bugs = dict(entries=[bug])
        [bug] = list(self.launchpad.bugs)
        self.assertEqual('<FakeEntry bug 1 at %s>' % hex(id(bug)), repr(bug))