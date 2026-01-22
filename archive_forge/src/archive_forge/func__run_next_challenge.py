from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import sys
from google_reauth import challenges
from google_reauth import errors
from google_reauth import _helpers
from google_reauth import _reauth_client
from six.moves import http_client
from six.moves import range
def _run_next_challenge(msg, http_request, access_token):
    """Get the next challenge from msg and run it.

    Args:
        msg: Reauth API response body (either from the initial request to
            https://reauth.googleapis.com/v2/sessions:start or from sending the
            previous challenge response to
            https://reauth.googleapis.com/v2/sessions/id:continue)
        http_request: callable to run http requests. Accepts uri, method, body
            and headers. Returns a tuple: (response, content)
        access_token: reauth access token

    Returns: rapt token.
    Raises:
        errors.ReauthError if reauth failed
    """
    for challenge in msg['challenges']:
        if challenge['status'] != 'READY':
            continue
        c = challenges.AVAILABLE_CHALLENGES.get(challenge['challengeType'], None)
        if not c:
            raise errors.ReauthFailError('Unsupported challenge type {0}. Supported types: {1}'.format(challenge['challengeType'], ','.join(list(challenges.AVAILABLE_CHALLENGES.keys()))))
        if not c.is_locally_eligible:
            raise errors.ReauthFailError('Challenge {0} is not locally eligible'.format(challenge['challengeType']))
        client_input = c.obtain_challenge_input(challenge)
        if not client_input:
            return None
        return _reauth_client.send_challenge_result(http_request, msg['sessionId'], challenge['challengeId'], client_input, access_token)
    return None