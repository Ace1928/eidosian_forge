import flask
import uuid
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.auth.plugins import mapped
import keystone.conf
from keystone import exception
from keystone.federation import utils as mapping_utils
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from unittest import mock
def _pull_assertion_from_the_request_headers(self):
    app = flask.Flask(__name__)
    with app.test_request_context(path='/path', environ_overrides=mapping_fixtures.UNICODE_NAME_ASSERTION):
        data = mapping_utils.get_assertion_params_from_env()
        return dict(data)