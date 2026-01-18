from unittest import mock
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavors
def test_set_with_valid_keys(self):
    valid_keys = ['key4', 'month.price', 'I-Am:AK-ey.44-', 'key with spaces and _']
    f = self.cs.flavors.get(4)
    for key in valid_keys:
        fk = f.set_keys({key: 'v4'})
        self.assert_request_id(fk, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('POST', '/flavors/4/os-extra_specs', {'extra_specs': {key: 'v4'}})