from __future__ import absolute_import, unicode_literals
import json
import logging
from oauthlib.common import Request
from ..errors import OAuth2Error, UnsupportedTokenTypeError
from .base import BaseEndpoint, catch_errors_and_unavailability
def validate_introspect_request(self, request):
    """Ensure the request is valid.

        The protected resource calls the introspection endpoint using
        an HTTP POST request with parameters sent as
        "application/x-www-form-urlencoded".

        token REQUIRED.  The string value of the token.

        token_type_hint OPTIONAL.
        A hint about the type of the token submitted for
        introspection.  The protected resource MAY pass this parameter to
        help the authorization server optimize the token lookup.  If the
        server is unable to locate the token using the given hint, it MUST
        extend its search across all of its supported token types.  An
        authorization server MAY ignore this parameter, particularly if it
        is able to detect the token type automatically.
            *  access_token: An Access Token as defined in [`RFC6749`],
                `section 1.4`_

            *  refresh_token: A Refresh Token as defined in [`RFC6749`],
                `section 1.5`_

        The introspection endpoint MAY accept other OPTIONAL
        parameters to provide further context to the query.  For
        instance, an authorization server may desire to know the IP
        address of the client accessing the protected resource to
        determine if the correct client is likely to be presenting the
        token.  The definition of this or any other parameters are
        outside the scope of this specification, to be defined by
        service documentation or extensions to this specification.

        .. _`section 1.4`: http://tools.ietf.org/html/rfc6749#section-1.4
        .. _`section 1.5`: http://tools.ietf.org/html/rfc6749#section-1.5
        .. _`RFC6749`: http://tools.ietf.org/html/rfc6749
        """
    self._raise_on_missing_token(request)
    self._raise_on_invalid_client(request)
    self._raise_on_unsupported_token(request)