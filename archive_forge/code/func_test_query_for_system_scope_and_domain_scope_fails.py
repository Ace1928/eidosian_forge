import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_query_for_system_scope_and_domain_scope_fails(self):
    path = '/role_assignments?scope.system=all&scope.domain.id=%(domain_id)s' % {'domain_id': self.domain_id}
    self.get(path, expected_status=http.client.BAD_REQUEST)