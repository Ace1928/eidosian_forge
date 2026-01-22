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
class OAuthNotificationTests(OAuth1Tests, test_notifications.BaseNotificationTest):

    def test_create_consumer(self):
        consumer_ref = self._create_single_consumer()
        self._assert_notify_sent(consumer_ref['id'], test_notifications.CREATED_OPERATION, 'OS-OAUTH1:consumer')
        self._assert_last_audit(consumer_ref['id'], test_notifications.CREATED_OPERATION, 'OS-OAUTH1:consumer', cadftaxonomy.SECURITY_ACCOUNT)

    def test_update_consumer(self):
        consumer_ref = self._create_single_consumer()
        update_ref = {'consumer': {'description': uuid.uuid4().hex}}
        PROVIDERS.oauth_api.update_consumer(consumer_ref['id'], update_ref)
        self._assert_notify_sent(consumer_ref['id'], test_notifications.UPDATED_OPERATION, 'OS-OAUTH1:consumer')
        self._assert_last_audit(consumer_ref['id'], test_notifications.UPDATED_OPERATION, 'OS-OAUTH1:consumer', cadftaxonomy.SECURITY_ACCOUNT)

    def test_delete_consumer(self):
        consumer_ref = self._create_single_consumer()
        PROVIDERS.oauth_api.delete_consumer(consumer_ref['id'])
        self._assert_notify_sent(consumer_ref['id'], test_notifications.DELETED_OPERATION, 'OS-OAUTH1:consumer')
        self._assert_last_audit(consumer_ref['id'], test_notifications.DELETED_OPERATION, 'OS-OAUTH1:consumer', cadftaxonomy.SECURITY_ACCOUNT)

    def test_oauth_flow_notifications(self):
        """Test to ensure notifications are sent for oauth tokens.

        This test is very similar to test_oauth_flow, however
        there are additional checks in this test for ensuring that
        notifications for request token creation, and access token
        creation/deletion are emitted.
        """
        consumer = self._create_single_consumer()
        consumer_id = consumer['id']
        consumer_secret = consumer['secret']
        self.consumer = {'key': consumer_id, 'secret': consumer_secret}
        self.assertIsNotNone(self.consumer['secret'])
        url, headers = self._create_request_token(self.consumer, self.project_id)
        content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        credentials = _urllib_parse_qs_text_keys(content.result)
        request_key = credentials['oauth_token'][0]
        request_secret = credentials['oauth_token_secret'][0]
        self.request_token = oauth1.Token(request_key, request_secret)
        self.assertIsNotNone(self.request_token.key)
        request_key_string = request_key.decode()
        self._assert_notify_sent(request_key_string, test_notifications.CREATED_OPERATION, 'OS-OAUTH1:request_token')
        self._assert_last_audit(request_key_string, test_notifications.CREATED_OPERATION, 'OS-OAUTH1:request_token', cadftaxonomy.SECURITY_CREDENTIAL)
        url = self._authorize_request_token(request_key)
        body = {'roles': [{'id': self.role_id}]}
        resp = self.put(url, body=body, expected_status=http.client.OK)
        self.verifier = resp.result['token']['oauth_verifier']
        self.assertTrue(all((i in base.VERIFIER_CHARS for i in self.verifier)))
        self.assertEqual(8, len(self.verifier))
        self.request_token.set_verifier(self.verifier)
        url, headers = self._create_access_token(self.consumer, self.request_token)
        content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
        credentials = _urllib_parse_qs_text_keys(content.result)
        access_key = credentials['oauth_token'][0]
        access_secret = credentials['oauth_token_secret'][0]
        self.access_token = oauth1.Token(access_key, access_secret)
        self.assertIsNotNone(self.access_token.key)
        access_key_string = access_key.decode()
        self._assert_notify_sent(access_key_string, test_notifications.CREATED_OPERATION, 'OS-OAUTH1:access_token')
        self._assert_last_audit(access_key_string, test_notifications.CREATED_OPERATION, 'OS-OAUTH1:access_token', cadftaxonomy.SECURITY_CREDENTIAL)
        resp = self.delete('/users/%(user)s/OS-OAUTH1/access_tokens/%(auth)s' % {'user': self.user_id, 'auth': self.access_token.key.decode()})
        self.assertResponseStatus(resp, http.client.NO_CONTENT)
        self._assert_notify_sent(access_key_string, test_notifications.DELETED_OPERATION, 'OS-OAUTH1:access_token')
        self._assert_last_audit(access_key_string, test_notifications.DELETED_OPERATION, 'OS-OAUTH1:access_token', cadftaxonomy.SECURITY_CREDENTIAL)