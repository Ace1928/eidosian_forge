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
def _do_http_request(self, method, resource, json=None):
    verify = self._verify_server
    headers = self._build_auth_headers()
    try:
        resp = method(resource, headers=headers, json=json, verify=verify)
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
    return resp