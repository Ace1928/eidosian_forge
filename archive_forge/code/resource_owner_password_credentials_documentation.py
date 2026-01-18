from __future__ import absolute_import, unicode_literals
import json
import logging
from .. import errors
from ..request_validator import RequestValidator
from .base import GrantTypeBase

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request

        The client makes a request to the token endpoint by adding the
        following parameters using the "application/x-www-form-urlencoded"
        format per Appendix B with a character encoding of UTF-8 in the HTTP
        request entity-body:

        grant_type
                REQUIRED.  Value MUST be set to "password".

        username
                REQUIRED.  The resource owner username.

        password
                REQUIRED.  The resource owner password.

        scope
                OPTIONAL.  The scope of the access request as described by
                `Section 3.3`_.

        If the client type is confidential or the client was issued client
        credentials (or assigned other authentication requirements), the
        client MUST authenticate with the authorization server as described
        in `Section 3.2.1`_.

        The authorization server MUST:

        o  require client authentication for confidential clients or for any
            client that was issued client credentials (or with other
            authentication requirements),

        o  authenticate the client if client authentication is included, and

        o  validate the resource owner password credentials using its
            existing password validation algorithm.

        Since this access token request utilizes the resource owner's
        password, the authorization server MUST protect the endpoint against
        brute force attacks (e.g., using rate-limitation or generating
        alerts).

        .. _`Section 3.3`: https://tools.ietf.org/html/rfc6749#section-3.3
        .. _`Section 3.2.1`: https://tools.ietf.org/html/rfc6749#section-3.2.1
        