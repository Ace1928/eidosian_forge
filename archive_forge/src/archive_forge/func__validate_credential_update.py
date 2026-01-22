import json
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _validate_credential_update(self, credential_id, credential):
    if credential.get('type', '').lower() == 'ec2' and (not credential.get('project_id')):
        existing_cred = self.get_credential(credential_id)
        if not existing_cred['project_id']:
            raise exception.ValidationError(attribute='project_id', target='credential')