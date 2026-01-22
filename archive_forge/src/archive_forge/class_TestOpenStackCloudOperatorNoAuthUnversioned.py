import openstack.cloud
from openstack.tests.unit import base
class TestOpenStackCloudOperatorNoAuthUnversioned(base.TestCase):

    def setUp(self):
        """Setup Noauth OpenStackCloud tests for unversioned endpoints

        Setup the test to utilize no authentication and an endpoint
        URL in the auth data.  This is permits testing of the basic
        mechanism that enables Ironic noauth mode to be utilized with
        Shade.

        Uses base.TestCase instead of IronicTestCase because
        we need to do completely different things with discovery.
        """
        super(TestOpenStackCloudOperatorNoAuthUnversioned, self).setUp()
        self._uri_registry.clear()
        self.register_uris([dict(method='GET', uri='https://baremetal.example.com/', json={'default_version': {'status': 'CURRENT', 'min_version': '1.1', 'version': '1.46', 'id': 'v1', 'links': [{'href': 'https://baremetal.example.com/v1', 'rel': 'self'}]}, 'versions': [{'status': 'CURRENT', 'min_version': '1.1', 'version': '1.46', 'id': 'v1', 'links': [{'href': 'https://baremetal.example.com/v1', 'rel': 'self'}]}], 'name': 'OpenStack Ironic API', 'description': 'Ironic is an OpenStack project.'}), dict(method='GET', uri=self.get_mock_url(service_type='baremetal', base_url_append='v1'), json={'media_types': [{'base': 'application/json', 'type': 'application/vnd.openstack.ironic.v1+json'}], 'links': [{'href': 'https://baremetal.example.com/v1', 'rel': 'self'}], 'ports': [{'href': 'https://baremetal.example.com/v1/ports/', 'rel': 'self'}, {'href': 'https://baremetal.example.com/ports/', 'rel': 'bookmark'}], 'nodes': [{'href': 'https://baremetal.example.com/v1/nodes/', 'rel': 'self'}, {'href': 'https://baremetal.example.com/nodes/', 'rel': 'bookmark'}], 'id': 'v1'}), dict(method='GET', uri=self.get_mock_url(service_type='baremetal', base_url_append='v1', resource='nodes'), json={'nodes': []})])

    def test_ironic_noauth_none_auth_type(self):
        """Test noauth selection for Ironic in OpenStackCloud

        The new way of doing this is with the keystoneauth none plugin.
        """
        self.cloud_noauth = openstack.connect(auth_type='none', baremetal_endpoint_override='https://baremetal.example.com')
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
        self.cloud_noauth = openstack.connect(auth_type='none', endpoint='https://baremetal.example.com/')
        self.cloud_noauth.list_machines()
        self.assert_calls()