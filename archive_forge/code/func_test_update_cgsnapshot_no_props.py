from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_update_cgsnapshot_no_props(self):
    cs.cgsnapshots.update('1234')