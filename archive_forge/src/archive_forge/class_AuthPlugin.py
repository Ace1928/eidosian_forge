import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
class AuthPlugin(plugin.BaseAuthPlugin):
    """Very simple debug authentication plugin.

    Takes Parameters such that it can throw exceptions at the right times.
    """
    TEST_TOKEN = utils.TestCase.TEST_TOKEN
    TEST_USER_ID = 'aUser'
    TEST_PROJECT_ID = 'aProject'
    SERVICE_URLS = {'identity': {'public': 'http://identity-public:1111/v2.0', 'admin': 'http://identity-admin:1111/v2.0'}, 'compute': {'public': 'http://compute-public:2222/v1.0', 'admin': 'http://compute-admin:2222/v1.0'}, 'image': {'public': 'http://image-public:3333/v2.0', 'admin': 'http://image-admin:3333/v2.0'}}

    def __init__(self, token=TEST_TOKEN, invalidate=True):
        self.token = token
        self._invalidate = invalidate

    def get_token(self, session):
        return self.token

    def get_endpoint(self, session, service_type=None, interface=None, **kwargs):
        try:
            return self.SERVICE_URLS[service_type][interface]
        except (KeyError, AttributeError):
            return None

    def invalidate(self):
        return self._invalidate

    def get_user_id(self, session):
        return self.TEST_USER_ID

    def get_project_id(self, session):
        return self.TEST_PROJECT_ID