from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def validate_client_key(self, client_key, request):
    """Validates that supplied client key is a registered and valid client.

        :param client_key: The client/consumer key.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :returns: True or False

        Note that if the dummy client is supplied it should validate in same
        or nearly the same amount of time as a valid one.

        Ensure latency inducing tasks are mimiced even for dummy clients.
        For example, use::

            from your_datastore import Client
            try:
                return Client.exists(client_key, access_token)
            except DoesNotExist:
                return False

        Rather than::

            from your_datastore import Client
            if access_token == self.dummy_access_token:
                return False
            else:
                return Client.exists(client_key, access_token)

        This method is used by

        * AccessTokenEndpoint
        * RequestTokenEndpoint
        * ResourceEndpoint
        * SignatureOnlyEndpoint
        """
    raise self._subclass_must_implement('validate_client_key')