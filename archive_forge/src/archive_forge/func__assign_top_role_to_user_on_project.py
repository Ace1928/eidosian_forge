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
def _assign_top_role_to_user_on_project(self, user, project):
    PROVIDERS.assignment_api.add_role_to_user_and_project(user['id'], project['id'], self.role_list[0]['id'])