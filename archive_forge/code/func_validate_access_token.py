from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def validate_access_token(self, client_key, token, request):
    """Validates that supplied access token is registered and valid.

        :param client_key: The client/consumer key.
        :param token: The access token string.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :returns: True or False

        Note that if the dummy access token is supplied it should validate in
        the same or nearly the same amount of time as a valid one.

        Ensure latency inducing tasks are mimiced even for dummy clients.
        For example, use::

            from your_datastore import AccessToken
            try:
                return AccessToken.exists(client_key, access_token)
            except DoesNotExist:
                return False

        Rather than::

            from your_datastore import AccessToken
            if access_token == self.dummy_access_token:
                return False
            else:
                return AccessToken.exists(client_key, access_token)

        This method is used by

        * ResourceEndpoint
        """
    raise self._subclass_must_implement('validate_access_token')