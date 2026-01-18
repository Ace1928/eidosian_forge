import json
import logging
from oauthlib.common import Request
from oauthlib.oauth2.rfc6749 import errors
from oauthlib.oauth2.rfc6749.endpoints.base import (
from oauthlib.oauth2.rfc6749.tokens import BearerToken
def validate_userinfo_request(self, request):
    """Ensure the request is valid.

        5.3.1.  UserInfo Request
        The Client sends the UserInfo Request using either HTTP GET or HTTP
        POST. The Access Token obtained from an OpenID Connect Authentication
        Request MUST be sent as a Bearer Token, per `Section 2`_ of OAuth 2.0
        Bearer Token Usage [RFC6750].

        It is RECOMMENDED that the request use the HTTP GET method and the
        Access Token be sent using the Authorization header field.

        The following is a non-normative example of a UserInfo Request:

        .. code-block:: http

            GET /userinfo HTTP/1.1
            Host: server.example.com
            Authorization: Bearer SlAV32hkKG

        5.3.3. UserInfo Error Response
        When an error condition occurs, the UserInfo Endpoint returns an Error
        Response as defined in `Section 3`_ of OAuth 2.0 Bearer Token Usage
        [RFC6750]. (HTTP errors unrelated to RFC 6750 are returned to the User
        Agent using the appropriate HTTP status code.)

        The following is a non-normative example of a UserInfo Error Response:

        .. code-block:: http

            HTTP/1.1 401 Unauthorized
            WWW-Authenticate: Bearer error="invalid_token",
                error_description="The Access Token expired"

        .. _`Section 2`: https://datatracker.ietf.org/doc/html/rfc6750#section-2
        .. _`Section 3`: https://datatracker.ietf.org/doc/html/rfc6750#section-3
        """
    if not self.bearer.validate_request(request):
        raise errors.InvalidTokenError()
    if 'openid' not in request.scopes:
        raise errors.InsufficientScopeError()