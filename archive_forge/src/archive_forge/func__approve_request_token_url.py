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
def _approve_request_token_url(self):
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
    url = self._authorize_request_token(request_key)
    return url