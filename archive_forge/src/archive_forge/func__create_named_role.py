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
def _create_named_role(self, name):
    role = unit.new_role_ref()
    role['name'] = name
    PROVIDERS.role_api.create_role(role['id'], role)
    return role