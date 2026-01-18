from unittest import mock
from openstack.tests.unit import base
def verify_get_overrided(self, proxy, resource_type, patch_target):
    with mock.patch(patch_target, autospec=True) as res:
        proxy._get_resource = mock.Mock(return_value=res)
        proxy._get(resource_type)
        res.fetch.assert_called_once_with(proxy, requires_id=True, base_path=None, error_message=mock.ANY, skip_cache=False)