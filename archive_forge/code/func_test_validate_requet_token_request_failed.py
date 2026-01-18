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
def test_validate_requet_token_request_failed(self):
    self.config_fixture.config(debug=True, insecure_debug=True)
    consumer = self._create_single_consumer()
    consumer_id = consumer['id']
    consumer_secret = consumer['secret']
    consumer = {'key': consumer_id, 'secret': consumer_secret}
    url = '/OS-OAUTH1/request_token'
    auth_header = 'OAuth oauth_version="1.0", oauth_consumer_key=' + consumer_id
    faked_header = {'Authorization': auth_header, 'requested_project_id': self.project_id}
    resp = self.post(url, headers=faked_header, expected_status=http.client.BAD_REQUEST)
    resp_data = jsonutils.loads(resp.body)
    self.assertIn('Validation failed with errors', resp_data['error']['message'])