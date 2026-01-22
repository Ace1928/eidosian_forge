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
def _build_effective_role_assignments_url(self, user):
    return '/role_assignments?effective&user.id=%(user_id)s' % {'user_id': user['id']}