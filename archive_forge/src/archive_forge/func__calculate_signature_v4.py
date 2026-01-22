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
def _calculate_signature_v4(string_to_sign, secret_key):
    """Calculate a v4 signature.

    :param bytes string_to_sign: String that contains request params and
                                 is used for calculate signature of request
    :param text secret_key: Second auth key of EC2 account that is used to
                            sign requests
    """
    parts = string_to_sign.split(b'\n')
    if len(parts) != 4 or parts[0] != b'AWS4-HMAC-SHA256':
        raise exception.Unauthorized(message=_('Invalid EC2 signature.'))
    scope = parts[2].split(b'/')
    if len(scope) != 4 or scope[3] != b'aws4_request':
        raise exception.Unauthorized(message=_('Invalid EC2 signature.'))
    allowed_services = [b's3', b'iam', b'sts']
    if scope[2] not in allowed_services:
        raise exception.Unauthorized(message=_('Invalid EC2 signature.'))

    def _sign(key, msg):
        return hmac.new(key, msg, hashlib.sha256).digest()
    signed = _sign(('AWS4' + secret_key).encode('utf-8'), scope[0])
    signed = _sign(signed, scope[1])
    signed = _sign(signed, scope[2])
    signed = _sign(signed, b'aws4_request')
    signature = hmac.new(signed, string_to_sign, hashlib.sha256)
    return signature.hexdigest()