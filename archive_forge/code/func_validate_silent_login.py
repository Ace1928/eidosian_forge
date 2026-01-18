from __future__ import absolute_import, unicode_literals
import logging
def validate_silent_login(self, request):
    """Ensure session user has authorized silent OpenID login.

        If no user is logged in or has not authorized silent login, this
        method should return False.

        If the user is logged in but associated with multiple accounts and
        not selected which one to link to the token then this method should
        raise an oauthlib.oauth2.AccountSelectionRequired error.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is used by:
            - OpenIDConnectAuthCode
            - OpenIDConnectImplicit
            - OpenIDConnectHybrid
        """
    raise NotImplementedError('Subclasses must implement this method.')