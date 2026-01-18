from __future__ import absolute_import, unicode_literals
import logging
from itertools import chain
from oauthlib.common import add_params_to_uri
from oauthlib.uri_validate import is_absolute_uri
from oauthlib.oauth2.rfc6749 import errors, utils
from ..request_validator import RequestValidator
def validate_grant_type(self, request):
    """
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        """
    client_id = getattr(request, 'client_id', None)
    if not self.request_validator.validate_grant_type(client_id, request.grant_type, request.client, request):
        log.debug('Unauthorized from %r (%r) access to grant type %s.', request.client_id, request.client, request.grant_type)
        raise errors.UnauthorizedClientError(request=request)