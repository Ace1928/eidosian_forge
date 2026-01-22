from openstack.cloud import _utils
from openstack import exceptions
from openstack.identity.v3._proxy import Proxy
from openstack import utils
def _get_domain_id_param_dict(self, domain_id):
    """Get a useable domain."""
    if utils.supports_version(self.identity, '3'):
        if not domain_id:
            raise exceptions.SDKException('User or project creation requires an explicit domain_id argument.')
        else:
            return {'domain_id': domain_id}
    else:
        return {}