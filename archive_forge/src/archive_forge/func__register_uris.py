import ddt
from keystoneauth1 import exceptions
from openstack.tests.unit import base
def _register_uris(self, status_code=None):
    uri = dict(method='GET', uri=self.get_mock_url('placement', 'public', append=['allocation_candidates']), json={})
    if status_code is not None:
        uri['status_code'] = status_code
    self.register_uris([uri])