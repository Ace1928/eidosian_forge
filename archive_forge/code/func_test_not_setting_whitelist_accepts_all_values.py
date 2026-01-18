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
def test_not_setting_whitelist_accepts_all_values(self):
    """Test that not setting whitelist passes.

        Not adding a ``whitelist`` keyword to the mapping value is different
        than adding empty whitelist.  The former case will simply pass all the
        values, whereas the latter would discard all the values.

        This test checks a scenario where a ``whitelist`` was not specified.
        Expected result is that no groups are ignored.

        The test scenario is as follows:
         - Create group ``EXISTS``
         - Set mapping rules for existing IdP with an empty whitelist
           that whould discard any values from the assertion
         - Issue an unscoped token and make sure ephemeral user is a member of
           two groups.

        """
    domain_id = self.domainA['id']
    domain_name = self.domainA['name']
    group_exists = unit.new_group_ref(domain_id=domain_id, name='EXISTS')
    group_exists = PROVIDERS.identity_api.create_group(group_exists)
    group_no_exists = unit.new_group_ref(domain_id=domain_id, name='NO_EXISTS')
    group_no_exists = PROVIDERS.identity_api.create_group(group_no_exists)
    group_ids = set([group_exists['id'], group_no_exists['id']])
    rules = {'rules': [{'local': [{'user': {'name': '{0}', 'id': '{0}'}}], 'remote': [{'type': 'REMOTE_USER'}]}, {'local': [{'groups': '{0}', 'domain': {'name': domain_name}}], 'remote': [{'type': 'REMOTE_USER_GROUPS'}]}]}
    PROVIDERS.federation_api.update_mapping(self.mapping['id'], rules)
    r = self._issue_unscoped_token(assertion='UNMATCHED_GROUP_ASSERTION')
    assigned_group_ids = r.federated_groups
    self.assertEqual(len(group_ids), len(assigned_group_ids))
    for group in assigned_group_ids:
        self.assertIn(group['id'], group_ids)