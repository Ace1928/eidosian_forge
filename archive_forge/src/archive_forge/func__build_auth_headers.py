import binascii
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.primitives.serialization import NoEncryption
from cryptography.hazmat.primitives.serialization import PrivateFormat
from cryptography.hazmat.primitives.serialization import PublicFormat
import os
import time
import uuid
from keystoneauth1 import loading
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import timeutils
import requests
from castellan.common import exception
from castellan.common.objects import private_key as pri_key
from castellan.common.objects import public_key as pub_key
from castellan.common.objects import symmetric_key as sym_key
from castellan.i18n import _
from castellan.key_manager import key_manager
def _build_auth_headers(self):
    if self._root_token_id:
        return self._set_namespace({'X-Vault-Token': self._root_token_id})
    if self._approle_token_id:
        return self._set_namespace({'X-Vault-Token': self._approle_token_id})
    if self._approle_role_id:
        params = {'role_id': self._approle_role_id}
        if self._approle_secret_id:
            params['secret_id'] = self._approle_secret_id
        approle_login_url = '{}v1/auth/approle/login'.format(self._get_url())
        token_issue_utc = timeutils.utcnow()
        headers = self._set_namespace({})
        try:
            resp = self._session.post(url=approle_login_url, json=params, headers=headers, verify=self._verify_server)
        except requests.exceptions.Timeout as ex:
            raise exception.KeyManagerError(str(ex))
        except requests.exceptions.ConnectionError as ex:
            raise exception.KeyManagerError(str(ex))
        except Exception as ex:
            raise exception.KeyManagerError(str(ex))
        if resp.status_code in _EXCEPTIONS_BY_CODE:
            raise exception.KeyManagerError(resp.reason)
        if resp.status_code == requests.codes['forbidden']:
            raise exception.Forbidden()
        resp_data = resp.json()
        if resp.status_code == requests.codes['bad_request']:
            raise exception.KeyManagerError(', '.join(resp_data['errors']))
        self._cached_approle_token_id = resp_data['auth']['client_token']
        self._approle_token_issue = token_issue_utc
        self._approle_token_ttl = resp_data['auth']['lease_duration']
        return self._set_namespace({'X-Vault-Token': self._approle_token_id})
    return {}