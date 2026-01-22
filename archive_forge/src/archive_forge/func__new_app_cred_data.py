import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def _new_app_cred_data(self, user_id, project_id=None, name=None, expires=None, system=None):
    if not name:
        name = uuid.uuid4().hex
    if not expires:
        expires = datetime.datetime.utcnow() + datetime.timedelta(days=365)
    if not system:
        system = uuid.uuid4().hex
    if not project_id:
        project_id = uuid.uuid4().hex
    app_cred_data = {'id': uuid.uuid4().hex, 'name': name, 'description': uuid.uuid4().hex, 'user_id': user_id, 'project_id': project_id, 'system': system, 'expires_at': expires, 'roles': [{'id': self.role__member_['id']}], 'secret': uuid.uuid4().hex, 'unrestricted': False}
    return app_cred_data