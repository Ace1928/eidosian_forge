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
def DoOneRoundOfChallenges(self, msg):
    next_msg = None
    for challenge in msg['challenges']:
        if challenge['status'] != 'READY':
            continue
        c = self.challenges[challenge['challengeType']]
        next_msg = c.Execute(challenge, msg['sessionId'])
    return next_msg