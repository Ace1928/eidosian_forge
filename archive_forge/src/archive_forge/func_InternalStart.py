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
def InternalStart(self, requested_scopes):
    """Does initial request to reauth API and initialize the challenges."""
    body = {'supportedChallengeTypes': list(self.challenges.keys())}
    if requested_scopes:
        body['oauthScopesForDomainPolicyLookup'] = requested_scopes
    _, content = self.http_request('{0}:start'.format(REAUTH_API), method='POST', body=json.dumps(body), headers={'Authorization': 'Bearer ' + self.access_token})
    response = json.loads(content)
    HandleErrors(response)
    return response