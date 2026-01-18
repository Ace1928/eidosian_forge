from unittest import mock
import warnings
from oslotest import base
from monascaclient import client
@mock.patch('monascaclient.client.migration')
@mock.patch('monascaclient.client._get_auth_handler')
@mock.patch('monascaclient.client._get_session')
def test_should_override_tenant_name_with_project_name(self, _, get_auth, __):
    api_version = mock.Mock()
    auth_val = mock.Mock()
    tenant_name = mock.Mock()
    project_name = tenant_name
    get_auth.return_value = auth_val
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        client.Client(api_version, tenant_name=tenant_name)
        self.assertEqual(1, len(w))
        self.assertEqual(DeprecationWarning, w[0].category)
        self.assertRegex(str(w[0].message), 'Usage of tenant_name has been deprecated in favour ')
    get_auth.assert_called_once_with({'project_name': project_name})