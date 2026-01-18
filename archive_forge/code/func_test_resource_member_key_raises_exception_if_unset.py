import uuid
import fixtures
import flask
import flask_restful
import functools
from oslo_policy import policy
from oslo_serialization import jsonutils
from testtools import matchers
from keystone.common import context
from keystone.common import json_home
from keystone.common import rbac_enforcer
import keystone.conf
from keystone import exception
from keystone.server.flask import common as flask_common
from keystone.server.flask.request_processing import json_body
from keystone.tests.unit import rest
def test_resource_member_key_raises_exception_if_unset(self):

    class TestResource(flask_common.ResourceBase):
        """A Test Resource."""

    class TestResourceWithKey(flask_common.ResourceBase):
        member_key = uuid.uuid4().hex
    r = TestResource()
    self.assertRaises(ValueError, getattr, r, 'member_key')
    r = TestResourceWithKey()
    self.assertEqual(TestResourceWithKey.member_key, r.member_key)