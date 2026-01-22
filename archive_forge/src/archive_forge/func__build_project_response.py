import fixtures
import uuid
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import projects
def _build_project_response(self, tags):
    project_id = uuid.uuid4().hex
    ret = {'projects': [{'is_domain': False, 'description': '', 'tags': tags, 'enabled': True, 'id': project_id, 'parent_id': 'default', 'domain_id': 'default', 'name': project_id}]}
    return ret