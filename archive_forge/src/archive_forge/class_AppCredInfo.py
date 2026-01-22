from oslo_log import log
from pycadf import cadftaxonomy as taxonomy
from pycadf import reason
from pycadf import resource
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
class AppCredInfo(BaseUserInfo):

    def __init__(self):
        super(AppCredInfo, self).__init__()
        self.id = None
        self.secret = None

    def _validate_and_normalize_auth_data(self, auth_payload):
        app_cred_api = PROVIDERS.application_credential_api
        if auth_payload.get('id'):
            app_cred = app_cred_api.get_application_credential(auth_payload['id'])
            self.user_id = app_cred['user_id']
            if not auth_payload.get('user'):
                auth_payload['user'] = {}
                auth_payload['user']['id'] = self.user_id
            super(AppCredInfo, self)._validate_and_normalize_auth_data(auth_payload)
        elif auth_payload.get('name'):
            super(AppCredInfo, self)._validate_and_normalize_auth_data(auth_payload)
            hints = driver_hints.Hints()
            hints.add_filter('name', auth_payload['name'])
            app_cred = app_cred_api.list_application_credentials(self.user_id, hints)[0]
            auth_payload['id'] = app_cred['id']
        else:
            raise exception.ValidationError(attribute='id or name', target='application credential')
        self.id = auth_payload['id']
        self.secret = auth_payload.get('secret')