import datetime
from testtools import matchers
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def _app_cred_body(self, roles=None, name=None, expires=None, secret=None, access_rules=None):
    name = name or uuid.uuid4().hex
    description = 'Credential for backups'
    app_cred_data = {'name': name, 'description': description}
    if roles:
        app_cred_data['roles'] = roles
    if expires:
        app_cred_data['expires_at'] = expires
    if secret:
        app_cred_data['secret'] = secret
    if access_rules is not None:
        app_cred_data['access_rules'] = access_rules
    return {'application_credential': app_cred_data}