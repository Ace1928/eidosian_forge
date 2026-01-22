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
class MappedResource(flask_restful.Resource):

    def post(self):
        rbac_enforcer.enforcer.RBACEnforcer().enforce_call(action='example:allowed')
        post_body = flask.request.get_json()
        return {'post_body': post_body}