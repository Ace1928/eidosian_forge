from __future__ import absolute_import, unicode_literals
import logging
def validate_response_type(self, client_id, response_type, client, request, *args, **kwargs):
    """Ensure client is authorized to use the response_type requested.

        :param client_id: Unicode client identifier.
        :param response_type: Unicode response type, i.e. code, token.
        :param client: Client object set by you, see ``.authenticate_client``.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is used by:
            - Authorization Code Grant
            - Implicit Grant
        """
    raise NotImplementedError('Subclasses must implement this method.')