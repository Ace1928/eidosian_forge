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
def _create_totp_cred(self):
    totp_cred = unit.new_totp_credential(self.user_id, self.project_id)
    PROVIDERS.credential_api.create_credential(uuid.uuid4().hex, totp_cred)

    def cleanup(testcase):
        totp_creds = testcase.credential_api.list_credentials_for_user(testcase.user['id'], type='totp')
        for cred in totp_creds:
            testcase.credential_api.delete_credential(cred['id'])
    self.addCleanup(cleanup, testcase=self)
    return totp_cred