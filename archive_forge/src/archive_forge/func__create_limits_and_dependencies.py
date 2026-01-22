import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def _create_limits_and_dependencies(domain_id=None):
    """Create limits and its dependencies for testing."""
    if not domain_id:
        domain_id = CONF.identity.default_domain_id
    service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
    registered_limit = unit.new_registered_limit_ref(service_id=service['id'], id=uuid.uuid4().hex)
    registered_limits = PROVIDERS.unified_limit_api.create_registered_limits([registered_limit])
    registered_limit = registered_limits[0]
    domain_limit = unit.new_limit_ref(domain_id=domain_id, service_id=service['id'], resource_name=registered_limit['resource_name'], resource_limit=10, id=uuid.uuid4().hex)
    project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=domain_id))
    project_limit = unit.new_limit_ref(project_id=project['id'], service_id=service['id'], resource_name=registered_limit['resource_name'], resource_limit=5, id=uuid.uuid4().hex)
    limits = PROVIDERS.unified_limit_api.create_limits([domain_limit, project_limit])
    project_limit_id = None
    domain_limit_id = None
    for limit in limits:
        if limit.get('domain_id'):
            domain_limit_id = limit['id']
        else:
            project_limit_id = limit['id']
    return (project_limit_id, domain_limit_id)