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
def _create_trust_get_token(self):
    ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.user_id, project_id=self.project_id, impersonation=True, expires=dict(minutes=1), role_ids=[self.role_id])
    del ref['id']
    r = self.post('/OS-TRUST/trusts', body={'trust': ref})
    trust = self.assertValidTrustResponse(r)
    auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], trust_id=trust['id'])
    return self.get_requested_token(auth_data)