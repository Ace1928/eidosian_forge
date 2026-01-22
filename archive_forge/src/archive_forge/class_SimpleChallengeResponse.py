from unittest import mock
import uuid
import stevedore
from keystone.api._shared import authentication
from keystone import auth
from keystone.auth.plugins import base
from keystone.auth.plugins import mapped
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit.ksfixtures import auth_plugins
class SimpleChallengeResponse(base.AuthMethodHandler):

    def authenticate(self, auth_payload):
        response_data = {}
        if 'response' in auth_payload:
            if auth_payload['response'] != EXPECTED_RESPONSE:
                raise exception.Unauthorized('Wrong answer')
            response_data['user_id'] = DEMO_USER_ID
            return base.AuthHandlerResponse(status=True, response_body=None, response_data=response_data)
        else:
            return base.AuthHandlerResponse(status=False, response_body={'challenge': "What's the name of your high school?"}, response_data=None)