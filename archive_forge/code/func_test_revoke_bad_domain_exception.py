import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_revoke_bad_domain_exception(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='domains', append=['baddomain']), status_code=404), dict(method='GET', uri=self.get_mock_url(resource='domains', qs_elements=['name=baddomain']), status_code=404)])
    with testtools.ExpectedException(exceptions.NotFoundException):
        self.cloud.revoke_role(self.role_data.role_name, user=self.user_data.name, domain='baddomain')
    self.assert_calls()