import hashlib
import flask
import http.client
from oslo_serialization import jsonutils
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone.credential import schema
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _validate_blob_update_keys(self, credential, ref):
    if credential.get('type', '').lower() == 'ec2':
        new_blob = self._validate_blob_json(ref)
        old_blob = credential.get('blob')
        if isinstance(old_blob, str):
            old_blob = jsonutils.loads(old_blob)
        for key in ['trust_id', 'app_cred_id', 'access_token_id', 'access_id']:
            if old_blob.get(key) != new_blob.get(key):
                message = _('%s can not be updated for credential') % key
                raise exception.ValidationError(message=message)