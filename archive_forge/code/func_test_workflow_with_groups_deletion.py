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
def test_workflow_with_groups_deletion(self):
    """Test full workflow with groups deletion before token scoping.

        The test scenario is as follows:
         - Create group ``group``
         - Create and assign roles to ``group`` and ``project_all``
         - Patch mapping rules for existing IdP so it issues group id
         - Issue unscoped token with ``group``'s id
         - Delete group ``group``
         - Scope token to ``project_all``
         - Expect HTTP 500 response

        """
    group = unit.new_group_ref(domain_id=self.domainA['id'])
    group = PROVIDERS.identity_api.create_group(group)
    role = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role['id'], role)
    PROVIDERS.assignment_api.create_grant(role['id'], group_id=group['id'], project_id=self.project_all['id'])
    rules = {'rules': [{'local': [{'group': {'id': group['id']}}, {'user': {'name': '{0}'}}], 'remote': [{'type': 'UserName'}, {'type': 'LastName', 'any_one_of': ['Account']}]}]}
    PROVIDERS.federation_api.update_mapping(self.mapping['id'], rules)
    r = self._issue_unscoped_token(assertion='TESTER_ASSERTION')
    PROVIDERS.identity_api.delete_group(group['id'])
    scoped_token = self._scope_request(r.id, 'project', self.project_all['id'])
    self.v3_create_token(scoped_token, expected_status=http.client.INTERNAL_SERVER_ERROR)