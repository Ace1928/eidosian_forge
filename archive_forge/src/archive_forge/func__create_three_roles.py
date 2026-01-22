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
def _create_three_roles(self):
    self.role_list = []
    for _ in range(3):
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        self.role_list.append(role)