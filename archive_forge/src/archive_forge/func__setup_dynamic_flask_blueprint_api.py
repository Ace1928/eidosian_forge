from unittest import mock
import uuid
import fixtures
import flask
from flask import blueprints
import flask_restful
from oslo_policy import policy
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import rest
def _setup_dynamic_flask_blueprint_api(self):
    api = uuid.uuid4().hex
    url_prefix = '/_%s_TEST' % api
    blueprint = blueprints.Blueprint(api, __name__, url_prefix=url_prefix)
    self.url_prefix = url_prefix
    self.flask_blueprint = blueprint
    self.cleanup_instance('flask_blueprint', 'url_prefix')