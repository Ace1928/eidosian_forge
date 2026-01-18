from __future__ import absolute_import, unicode_literals
import logging
def validate_bearer_token(self, token, scopes, request):
    """Ensure the Bearer token is valid and authorized access to scopes.

        :param token: A string of random characters.
        :param scopes: A list of scopes associated with the protected resource.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request

        A key to OAuth 2 security and restricting impact of leaked tokens is
        the short expiration time of tokens, *always ensure the token has not
        expired!*.

        Two different approaches to scope validation:

            1) all(scopes). The token must be authorized access to all scopes
                            associated with the resource. For example, the
                            token has access to ``read-only`` and ``images``,
                            thus the client can view images but not upload new.
                            Allows for fine grained access control through
                            combining various scopes.

            2) any(scopes). The token must be authorized access to one of the
                            scopes associated with the resource. For example,
                            token has access to ``read-only-images``.
                            Allows for fine grained, although arguably less
                            convenient, access control.

        A powerful way to use scopes would mimic UNIX ACLs and see a scope
        as a group with certain privileges. For a restful API these might
        map to HTTP verbs instead of read, write and execute.

        Note, the request.user attribute can be set to the resource owner
        associated with this token. Similarly the request.client and
        request.scopes attribute can be set to associated client object
        and authorized scopes. If you then use a decorator such as the
        one provided for django these attributes will be made available
        in all protected views as keyword arguments.

        :param token: Unicode Bearer token
        :param scopes: List of scopes (defined by you)
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is indirectly used by all core Bearer token issuing grant types:
            - Authorization Code Grant
            - Implicit Grant
            - Resource Owner Password Credentials Grant
            - Client Credentials Grant
        """
    raise NotImplementedError('Subclasses must implement this method.')