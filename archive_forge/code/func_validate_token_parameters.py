from __future__ import absolute_import, unicode_literals
import json
import os
import time
from oauthlib.common import add_params_to_qs, add_params_to_uri, unicode_type
from oauthlib.signals import scope_changed
from .errors import (InsecureTransportError, MismatchingStateError,
from .tokens import OAuth2Token
from .utils import is_secure_transport, list_to_scope, scope_to_list
def validate_token_parameters(params):
    """Ensures token precence, token type, expiration and scope in params."""
    if 'error' in params:
        raise_from_error(params.get('error'), params)
    if not 'access_token' in params:
        raise MissingTokenError(description='Missing access token parameter.')
    if not 'token_type' in params:
        if os.environ.get('OAUTHLIB_STRICT_TOKEN_TYPE'):
            raise MissingTokenTypeError()
    if params.scope_changed:
        message = 'Scope has changed from "{old}" to "{new}".'.format(old=params.old_scope, new=params.scope)
        scope_changed.send(message=message, old=params.old_scopes, new=params.scopes)
        if not os.environ.get('OAUTHLIB_RELAX_TOKEN_SCOPE', None):
            w = Warning(message)
            w.token = params
            w.old_scope = params.old_scopes
            w.new_scope = params.scopes
            raise w