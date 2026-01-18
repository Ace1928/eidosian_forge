from cinderclient import api_versions
from cinderclient import exceptions as exc
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import group_types
def test_set_key_pre_version(self):
    t = group_types.GroupType(pre_cs, {'id': 1})
    self.assertRaises(exc.VersionNotFoundForAPIMethod, t.set_keys, {'k': 'v'})