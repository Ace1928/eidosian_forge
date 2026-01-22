import openstack.cloud
from openstack.tests.unit import base
class TestOpenStackCloudOperatorNoAuth(base.TestCase):

    def setUp(self):
        """Setup Noauth OpenStackCloud tests

        Setup the test to utilize no authentication and an endpoint
        URL in the auth data.  This is permits testing of the basic
        mechanism that enables Ironic noauth mode to be utilized with
        Shade.

        Uses base.TestCase instead of IronicTestCase because
        we need to do completely different things with discovery.
        """
        super(TestOpenStackCloudOperatorNoAuth, self).setUp()
        self._uri_registry.clear()
        self.register_uris([dict(method='GET', uri=self.get_mock_url(service_type='baremetal', base_url_append='v1'), json={'id': 'v1', 'links': [{'href': 'https://baremetal.example.com/v1', 'rel': 'self'}]}), dict(method='GET', uri=self.get_mock_url(service_type='baremetal', base_url_append='v1', resource='nodes'), json={'nodes': []})])

    def test_ironic_noauth_none_auth_type(self):
        """Test noauth selection for Ironic in OpenStackCloud

        The new way of doing this is with the keystoneauth none plugin.
        """
        self.cloud_noauth = openstack.connect(auth_type='none', baremetal_endpoint_override='https://baremetal.example.com/v1')
        self.cloud_noauth.list_machines()
        self.assert_calls()

    def test_ironic_noauth_auth_endpoint(self):
        """Test noauth selection for Ironic in OpenStackCloud

        Sometimes people also write clouds.yaml files that look like this:

        ::
          clouds:
            bifrost:
              auth_type: "none"
              endpoint: https://baremetal.example.com
        """
        self.cloud_noauth = openstack.connect(auth_type='none', endpoint='https://baremetal.example.com/v1')
        self.cloud_noauth.list_machines()
        self.assert_calls()

    def test_ironic_noauth_admin_token_auth_type(self):
        """Test noauth selection for Ironic in OpenStackCloud

        The old way of doing this was to abuse admin_token.
        """
        self.cloud_noauth = openstack.connect(auth_type='admin_token', auth=dict(endpoint='https://baremetal.example.com/v1', token='ignored'))
        self.cloud_noauth.list_machines()
        self.assert_calls()