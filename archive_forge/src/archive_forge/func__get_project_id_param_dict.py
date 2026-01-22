from openstack.cloud import _utils
from openstack import exceptions
from openstack.identity.v3._proxy import Proxy
from openstack import utils
def _get_project_id_param_dict(self, name_or_id):
    if name_or_id:
        project = self.get_project(name_or_id)
        if not project:
            return {}
        if utils.supports_version(self.identity, '3'):
            return {'default_project_id': project['id']}
        else:
            return {'tenant_id': project['id']}
    else:
        return {}