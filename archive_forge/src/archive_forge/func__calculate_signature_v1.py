import base64
import hashlib
import hmac
import flask
import http.client
from oslo_serialization import jsonutils
from keystone.api._shared import EC2_S3_Resource
from keystone.api._shared import json_home_relations
from keystone.common import render_token
from keystone.common import utils
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _calculate_signature_v1(string_to_sign, secret_key):
    """Calculate a v1 signature.

    :param bytes string_to_sign: String that contains request params and
                                 is used for calculate signature of request
    :param text secret_key: Second auth key of EC2 account that is used to
                            sign requests
    """
    key = str(secret_key).encode('utf-8')
    b64_encode = base64.encodebytes
    signed = b64_encode(hmac.new(key, string_to_sign, hashlib.sha1).digest()).decode('utf-8').strip()
    return signed