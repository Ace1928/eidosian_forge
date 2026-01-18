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
def test_oauth_token_cannot_authorize_request_token(self):
    self.test_oauth_flow()
    url = self._approve_request_token_url()
    body = {'roles': [{'id': self.role_id}]}
    self.put(url, body=body, token=self.keystone_token_id, expected_status=http.client.FORBIDDEN)