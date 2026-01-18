from __future__ import absolute_import, unicode_literals
import logging
from itertools import chain
from oauthlib.common import add_params_to_uri
from oauthlib.uri_validate import is_absolute_uri
from oauthlib.oauth2.rfc6749 import errors, utils
from ..request_validator import RequestValidator
def validate_scopes(self, request):
    """
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        """
    if not request.scopes:
        request.scopes = utils.scope_to_list(request.scope) or utils.scope_to_list(self.request_validator.get_default_scopes(request.client_id, request))
    log.debug('Validating access to scopes %r for client %r (%r).', request.scopes, request.client_id, request.client)
    if not self.request_validator.validate_scopes(request.client_id, request.scopes, request.client, request):
        raise errors.InvalidScopeError(request=request)