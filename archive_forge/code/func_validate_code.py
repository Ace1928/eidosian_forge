from __future__ import absolute_import, unicode_literals
import logging
def validate_code(self, client_id, code, client, request, *args, **kwargs):
    """Verify that the authorization_code is valid and assigned to the given

        client.

        Before returning true, set the following based on the information stored
        with the code in 'save_authorization_code':

            - request.user
            - request.state (if given)
            - request.scopes
            - request.claims (if given)
        OBS! The request.user attribute should be set to the resource owner
        associated with this authorization code. Similarly request.scopes
        must also be set.

        The request.claims property, if it was given, should assigned a dict.

        If PKCE is enabled (see 'is_pkce_required' and
        'save_authorization_code')
        you MUST set the following based on the information stored:
            - request.code_challenge
            - request.code_challenge_method

        :param client_id: Unicode client identifier.
        :param code: Unicode authorization code.
        :param client: Client object set by you, see ``.authenticate_client``.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is used by:
            - Authorization Code Grant
        """
    raise NotImplementedError('Subclasses must implement this method.')