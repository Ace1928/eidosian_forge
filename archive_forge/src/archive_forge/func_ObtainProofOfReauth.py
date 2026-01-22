import base64
import getpass
import json
import logging
import sys
from oauth2client.contrib import reauth_errors
from pyu2f import errors as u2ferrors
from pyu2f import model
from pyu2f.convenience import authenticator
from six.moves import urllib
def ObtainProofOfReauth(self, requested_scopes=None):
    """Obtain proof of reauth (rapt token)."""
    msg = None
    max_challenge_count = 5
    while max_challenge_count:
        max_challenge_count -= 1
        if not msg:
            msg = self.InternalStart(requested_scopes)
        if msg['status'] == 'AUTHENTICATED':
            return msg['encodedProofOfReauthToken']
        if not (msg['status'] == 'CHALLENGE_REQUIRED' or msg['status'] == 'CHALLENGE_PENDING'):
            raise reauth_errors.ReauthAPIError('Challenge status {0}'.format(msg['status']))
        if not InteractiveCheck():
            raise reauth_errors.ReauthUnattendedError()
        msg = self.DoOneRoundOfChallenges(msg)
    raise reauth_errors.ReauthFailError()