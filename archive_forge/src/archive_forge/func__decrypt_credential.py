import json
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _decrypt_credential(self, credential):
    """Return a decrypted credential reference."""
    if credential['type'] == 'ec2':
        decrypted_blob = json.loads(PROVIDERS.credential_provider_api.decrypt(credential['encrypted_blob']))
    else:
        decrypted_blob = PROVIDERS.credential_provider_api.decrypt(credential['encrypted_blob'])
    credential['blob'] = decrypted_blob
    credential.pop('key_hash', None)
    credential.pop('encrypted_blob', None)
    return credential