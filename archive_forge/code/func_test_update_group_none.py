import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_update_group_none(self):
    self.assertIsNone(cs.groups.update('1234'))