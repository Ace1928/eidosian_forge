import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def _list_ids(self, user):
    hints = driver_hints.Hints()
    resp = self.app_cred_api.list_application_credentials(user['id'], hints)
    return [ac['id'] for ac in resp]