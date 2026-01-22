from openstack.cloud import _utils
from openstack import exceptions
from openstack.identity.v3._proxy import Proxy
from openstack import utils
def _get_identity_params(self, domain_id=None, project=None):
    """Get the domain and project/tenant parameters if needed.

        keystone v2 and v3 are divergent enough that we need to pass or not
        pass project or tenant_id or domain or nothing in a sane manner.
        """
    ret = {}
    ret.update(self._get_domain_id_param_dict(domain_id))
    ret.update(self._get_project_id_param_dict(project))
    return ret