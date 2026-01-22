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
def _check_projects_and_roles(self, token, roles, projects):
    """Check whether the projects and the roles match."""
    token_roles = token.get('roles')
    if token_roles is None:
        raise AssertionError('Roles not found in the token')
    token_roles = self._roles(token_roles)
    roles_ref = self._roles(roles)
    self.assertEqual(token_roles, roles_ref)
    token_projects = token.get('project')
    if token_projects is None:
        raise AssertionError('Projects not found in the token')
    token_projects = self._project(token_projects)
    projects_ref = self._project(projects)
    self.assertEqual(token_projects, projects_ref)