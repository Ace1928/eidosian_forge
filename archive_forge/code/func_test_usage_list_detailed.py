import datetime
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import usage
def test_usage_list_detailed(self):
    self.test_usage_list(True)