from __future__ import absolute_import, unicode_literals
import logging
def validate_silent_authorization(self, request):
    """Ensure the logged in user has authorized silent OpenID authorization.

        Silent OpenID authorization allows access tokens and id tokens to be
        granted to clients without any user prompt or interaction.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is used by:
            - OpenIDConnectAuthCode
            - OpenIDConnectImplicit
            - OpenIDConnectHybrid
        """
    raise NotImplementedError('Subclasses must implement this method.')