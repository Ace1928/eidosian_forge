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
def test_oauth_token_cannot_create_new_trust(self):
    self.test_oauth_flow()
    ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.user_id, project_id=self.project_id, impersonation=True, expires=dict(minutes=1), role_ids=[self.role_id])
    del ref['id']
    self.post('/OS-TRUST/trusts', body={'trust': ref}, token=self.keystone_token_id, expected_status=http.client.FORBIDDEN)