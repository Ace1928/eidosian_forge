from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def _on_project_or_endpoint_delete(self, service, resource_type, operation, payload):
    project_or_endpoint_id = payload['resource_info']
    if resource_type == 'project':
        PROVIDERS.catalog_api.delete_association_by_project(project_or_endpoint_id)
        PROVIDERS.catalog_api.delete_endpoint_group_association_by_project(project_or_endpoint_id)
    else:
        PROVIDERS.catalog_api.delete_association_by_endpoint(project_or_endpoint_id)