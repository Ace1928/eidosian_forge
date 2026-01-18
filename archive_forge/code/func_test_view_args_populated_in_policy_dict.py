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
def test_view_args_populated_in_policy_dict(self):

    def _enforce_mock_func(credentials, action, target, do_raise=True):
        if 'argument_id' not in target:
            raise exception.ForbiddenAction(action=action)
    self.useFixture(fixtures.MockPatchObject(self.enforcer, '_enforce', _enforce_mock_func))
    argument_id = uuid.uuid4().hex
    with self.test_client() as c:
        path = '/v3/auth/tokens'
        body = self._auth_json()
        r = c.post(path, json=body, follow_redirects=True, expected_status_code=201)
        token_id = r.headers['X-Subject-Token']
        c.get('%s/argument/%s' % (self.restful_api_url_prefix, argument_id), headers={'X-Auth-Token': token_id})
        self.enforcer.enforce_call(action='example:allowed')
        c.get('%s/argument' % self.restful_api_url_prefix, headers={'X-Auth-Token': token_id})
        self.assertRaises(exception.ForbiddenAction, self.enforcer.enforce_call, action='example:allowed')