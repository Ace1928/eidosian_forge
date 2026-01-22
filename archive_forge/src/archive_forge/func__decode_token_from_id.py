import datetime
import os
import jwt
from oslo_utils import timeutils
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.token.providers import base
def _decode_token_from_id(self, token_id):
    options = dict()
    options['verify_exp'] = False
    for public_key in self.public_keys:
        try:
            return jwt.decode(token_id, public_key, algorithms=JWSFormatter.algorithm, options=options)
        except (jwt.InvalidSignatureError, jwt.DecodeError):
            pass
    raise exception.TokenNotFound(token_id=token_id)