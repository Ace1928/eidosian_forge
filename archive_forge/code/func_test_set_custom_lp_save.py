from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_set_custom_lp_save(self):
    """
        A custom C{lp_save} method can be set on a L{FakeResource} after its
        been created.
        """
    self.launchpad.me = dict(name='foo')
    self.launchpad.me.lp_save = lambda: 'custom'
    self.assertEqual('custom', self.launchpad.me.lp_save())