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
class S3Resource(EC2_S3_Resource.ResourceBase):

    @staticmethod
    def _check_signature(creds_ref, credentials):
        string_to_sign = base64.urlsafe_b64decode(str(credentials['token']))
        if string_to_sign[0:4] != b'AWS4':
            signature = _calculate_signature_v1(string_to_sign, creds_ref['secret'])
        else:
            signature = _calculate_signature_v4(string_to_sign, creds_ref['secret'])
        if not utils.auth_str_equal(credentials['signature'], signature):
            raise exception.Unauthorized(message=_('Credential signature mismatch'))

    @ks_flask.unenforced_api
    def post(self):
        """Authenticate s3token.

        POST /v3/s3tokens
        """
        token = self.handle_authenticate()
        token_reference = render_token.render_token_response_from_model(token)
        resp_body = jsonutils.dumps(token_reference)
        response = flask.make_response(resp_body, http.client.OK)
        response.headers['Content-Type'] = 'application/json'
        return response