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
def _store_key_value(self, key_id, value):
    type_value = self._secret_type_dict.get(type(value))
    if type_value is None:
        raise exception.KeyManagerError('Unknown type for value : %r' % value)
    record = {'type': type_value, 'value': binascii.hexlify(value.get_encoded()).decode('utf-8'), 'algorithm': value.algorithm if hasattr(value, 'algorithm') else None, 'bit_length': value.bit_length if hasattr(value, 'bit_length') else None, 'name': value.name, 'created': value.created}
    if self._kv_version > 1:
        record = {'data': record}
    self._do_http_request(self._session.post, self._get_resource_url(key_id), json=record)
    return key_id