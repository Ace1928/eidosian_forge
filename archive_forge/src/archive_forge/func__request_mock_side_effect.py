import base64
import json
import os
import unittest
import mock
from six.moves import http_client
from six.moves import urllib
from oauth2client import client
from google_reauth import challenges
from google_reauth import reauth
from google_reauth import errors
from google_reauth import reauth_creds
from google_reauth import _reauth_client
from google_reauth.reauth_creds import Oauth2WithReauthCredentials
from pyu2f import model
from pyu2f import u2f
def _request_mock_side_effect(self, *args, **kwargs):
    """Helper function to respond with valid requests as if a real server.

        This is the helper function for mocking HTTP calls. The calls that should
        end up here are to the oauth2 API or to the reauth API. The order of ifs
        tries to mimic the real order that the requests are expected, but we do not
        enforce a particular order so it can be more general.

        Args:
          *args: Every arg passed to a request.
          **kwargs: Every keyed arg passed to a request.

        Returns:
          (str, str), Mocked (headers, content)

        Raises:
          Exception: In case this function doesn't know how to mock a request.
        """
    qp = dict(urllib.parse.parse_qsl(kwargs['body']))
    try:
        qp_json = json.loads(kwargs['body'])
    except ValueError:
        qp_json = {}
    uri = kwargs['uri'] if 'uri' in kwargs else args[0]
    if uri == self.oauth_api_url and qp.get('scope') == reauth._REAUTH_SCOPE:
        return (_ok_response, json.dumps({'access_token': 'access_token_for_reauth'}))
    if uri == _reauth_client._REAUTH_API + ':start':
        return (None, json.dumps({'status': 'CHALLENGE_REQUIRED', 'sessionId': 'session_id_1', 'challenges': [{'status': 'READY', 'challengeId': 1, 'challengeType': 'PASSWORD', 'securityKey': {}}]}))
    if uri == _reauth_client._REAUTH_API + '/session_id_1:continue':
        self.assertEqual(1, qp_json.get('challengeId'))
        self.assertEqual('RESPOND', qp_json.get('action'))
        if qp_json.get('proposalResponse', {}).get('credential') == self.correct_password:
            return (None, json.dumps({'status': 'CHALLENGE_REQUIRED', 'sessionId': 'session_id_2', 'challenges': [{'status': 'READY', 'challengeId': 2, 'challengeType': 'SECURITY_KEY', 'securityKey': {'applicationId': 'security_key_application_id', 'challenges': [{'keyHandle': 'some_key', 'challenge': base64.urlsafe_b64encode('some_challenge'.encode('ascii')).decode('ascii')}]}}]}))
        else:
            return (None, json.dumps({'status': 'CHALLENGE_PENDING', 'sessionId': 'session_id_1', 'challenges': [{'status': 'READY', 'challengeId': 1, 'challengeType': 'PASSWORD', 'securityKey': {}}]}))
    if uri == _reauth_client._REAUTH_API + '/session_id_2:continue':
        self.assertEqual(2, qp_json.get('challengeId'))
        self.assertEqual('RESPOND', qp_json.get('action'))
        return (None, json.dumps({'status': 'AUTHENTICATED', 'sessionId': 'session_id_3', 'encodedProofOfReauthToken': self.rapt_token}))
    raise Exception('Unexpected call :/\nURL {0}\n{1}'.format(uri, kwargs['body']))