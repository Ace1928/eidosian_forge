from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_top_level_collection_property(self):
    """
        Sample top-level collections can be set on L{FakeLaunchpad}
        instances.  They are validated the same way other sample data is
        validated.
        """
    branch = dict(name='foo')
    self.launchpad.branches = dict(getByUniqueName=lambda name: branch)
    branch = self.launchpad.branches.getByUniqueName('foo')
    self.assertEqual('foo', branch.name)