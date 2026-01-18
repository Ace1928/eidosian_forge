import copy
import os
import random
import re
import subprocess
from testtools import matchers
from unittest import mock
import uuid
import fixtures
import flask
import http.client
from lxml import etree
from oslo_serialization import jsonutils
from oslo_utils import importutils
import saml2
from saml2 import saml
from saml2 import sigver
import urllib
from keystone.api._shared import authentication
from keystone.api import auth as auth_api
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import render_token
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.models import token_model
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import core
from keystone.tests.unit import federation_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_user_role_assignment(self):
    project_ref = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
    role_ref = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
    user_id, unscoped_token = self._authenticate_via_saml()
    v3_scope_request = self._scope_request(unscoped_token, 'project', project_ref['id'])
    r = self.v3_create_token(v3_scope_request, expected_status=http.client.UNAUTHORIZED)
    PROVIDERS.assignment_api.add_role_to_user_and_project(user_id, project_ref['id'], role_ref['id'])
    r = self.v3_create_token(v3_scope_request, expected_status=http.client.CREATED)
    scoped_token = r.headers['X-Subject-Token']
    path = '/projects/%(project_id)s' % {'project_id': project_ref['id']}
    r = self.v3_request(path=path, method='GET', expected_status=http.client.OK, token=scoped_token)
    self.assertValidProjectResponse(r, project_ref)
    project_ref2 = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    PROVIDERS.resource_api.create_project(project_ref2['id'], project_ref2)
    path = '/projects/%(project_id)s' % {'project_id': project_ref2['id']}
    r = self.v3_request(path=path, method='GET', expected_status=http.client.FORBIDDEN, token=scoped_token)