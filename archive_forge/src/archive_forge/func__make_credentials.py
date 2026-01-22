import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
def _make_credentials(self, cred_type, count=1, user_id=None, project_id=None, blob=None):
    user_id = user_id or self.default_domain_user['id']
    project_id = project_id or self.default_domain_project['id']
    creds = []
    for __ in range(count):
        if cred_type == 'totp':
            ref = unit.new_totp_credential(user_id=user_id, project_id=project_id, blob=blob)
        else:
            ref = unit.new_credential_ref(user_id=user_id, project_id=project_id)
        resp = self.post('/credentials', body={'credential': ref})
        creds.append(resp.json['credential'])
    return creds