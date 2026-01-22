import copy
import datetime
import random
from unittest import mock
import uuid
import freezegun
import http.client
from oslo_serialization import jsonutils
from pycadf import cadftaxonomy
import urllib
from urllib import parse as urlparse
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import oauth1
from keystone.oauth1.backends import base
from keystone.tests import unit
from keystone.tests.unit.common import test_notifications
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
class AccessTokenCRUDTests(OAuthFlowTests):

    def test_delete_access_token_dne(self):
        self.delete('/users/%(user)s/OS-OAUTH1/access_tokens/%(auth)s' % {'user': self.user_id, 'auth': uuid.uuid4().hex}, expected_status=http.client.NOT_FOUND)

    def test_list_no_access_tokens(self):
        url = '/users/%(user_id)s/OS-OAUTH1/access_tokens' % {'user_id': self.user_id}
        resp = self.get(url)
        entities = resp.result['access_tokens']
        self.assertEqual([], entities)
        self.assertValidListLinks(resp.result['links'])
        self.head(url, expected_status=http.client.OK)

    def test_get_single_access_token(self):
        self.test_oauth_flow()
        access_token_key_string = self.access_token.key.decode()
        url = '/users/%(user_id)s/OS-OAUTH1/access_tokens/%(key)s' % {'user_id': self.user_id, 'key': access_token_key_string}
        resp = self.get(url)
        entity = resp.result['access_token']
        self.assertEqual(access_token_key_string, entity['id'])
        self.assertEqual(self.consumer['key'], entity['consumer_id'])
        self.assertEqual('http://localhost/v3' + url, entity['links']['self'])
        self.head(url, expected_status=http.client.OK)

    def test_get_access_token_dne(self):
        url = '/users/%(user_id)s/OS-OAUTH1/access_tokens/%(key)s' % {'user_id': self.user_id, 'key': uuid.uuid4().hex}
        self.get(url, expected_status=http.client.NOT_FOUND)
        self.head(url, expected_status=http.client.NOT_FOUND)

    def test_list_all_roles_in_access_token(self):
        self.test_oauth_flow()
        url = '/users/%(id)s/OS-OAUTH1/access_tokens/%(key)s/roles' % {'id': self.user_id, 'key': self.access_token.key.decode()}
        resp = self.get(url)
        entities = resp.result['roles']
        self.assertTrue(entities)
        self.assertValidListLinks(resp.result['links'])
        self.head(url, expected_status=http.client.OK)

    def test_get_role_in_access_token(self):
        self.test_oauth_flow()
        access_token_key = self.access_token.key.decode()
        url = '/users/%(id)s/OS-OAUTH1/access_tokens/%(key)s/roles/%(role)s' % {'id': self.user_id, 'key': access_token_key, 'role': self.role_id}
        resp = self.get(url)
        entity = resp.result['role']
        self.assertEqual(self.role_id, entity['id'])
        self.head(url, expected_status=http.client.OK)

    def test_get_role_in_access_token_dne(self):
        self.test_oauth_flow()
        access_token_key = self.access_token.key.decode()
        url = '/users/%(id)s/OS-OAUTH1/access_tokens/%(key)s/roles/%(role)s' % {'id': self.user_id, 'key': access_token_key, 'role': uuid.uuid4().hex}
        self.get(url, expected_status=http.client.NOT_FOUND)
        self.head(url, expected_status=http.client.NOT_FOUND)

    def test_list_and_delete_access_tokens(self):
        self.test_oauth_flow()
        url = '/users/%(user_id)s/OS-OAUTH1/access_tokens' % {'user_id': self.user_id}
        resp = self.get(url)
        self.head(url, expected_status=http.client.OK)
        entities = resp.result['access_tokens']
        self.assertTrue(entities)
        self.assertValidListLinks(resp.result['links'])
        access_token_key = self.access_token.key.decode()
        resp = self.delete('/users/%(user)s/OS-OAUTH1/access_tokens/%(auth)s' % {'user': self.user_id, 'auth': access_token_key})
        self.assertResponseStatus(resp, http.client.NO_CONTENT)
        resp = self.get(url)
        self.head(url, expected_status=http.client.OK)
        entities = resp.result['access_tokens']
        self.assertEqual([], entities)
        self.assertValidListLinks(resp.result['links'])