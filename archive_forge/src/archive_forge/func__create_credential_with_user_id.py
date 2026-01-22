import uuid
from oslo_config import fixture as config_fixture
from keystone.common import provider_api
from keystone.credential.providers import fernet as credential_provider
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.credential.backends import sql as credential_sql
from keystone import exception
def _create_credential_with_user_id(self, user_id=None):
    if not user_id:
        user_id = uuid.uuid4().hex
    credential = unit.new_credential_ref(user_id=user_id, extra=uuid.uuid4().hex, type=uuid.uuid4().hex)
    PROVIDERS.credential_api.create_credential(credential['id'], credential)
    return credential