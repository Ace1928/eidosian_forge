import json
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _encrypt_credential(self, credential):
    """Return an encrypted credential reference."""
    credential_copy = credential.copy()
    if credential.get('type', None) == 'ec2':
        encrypted_blob, key_hash = PROVIDERS.credential_provider_api.encrypt(json.dumps(credential['blob']))
    else:
        encrypted_blob, key_hash = PROVIDERS.credential_provider_api.encrypt(credential['blob'])
    credential_copy['encrypted_blob'] = encrypted_blob
    credential_copy['key_hash'] = key_hash
    credential_copy.pop('blob', None)
    return credential_copy