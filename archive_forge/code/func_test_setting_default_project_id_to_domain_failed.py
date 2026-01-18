import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_log import log
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_setting_default_project_id_to_domain_failed(self):
    """Call ``POST and PATCH /users`` default_project_id=domain_id.

        Make sure we validate the default_project_id if it is specified.
        It cannot be set to a domain_id, even for a project acting as domain
        right now. That's because we haven't sort out the issuing
        project-scoped token for project acting as domain bit yet. Once we
        got that sorted out, we can relax this constraint.

        """
    ref = unit.new_user_ref(domain_id=self.domain_id, project_id=self.domain_id)
    self.post('/users', body={'user': ref}, token=CONF.admin_token, expected_status=http.client.BAD_REQUEST)
    user = {'default_project_id': self.domain_id}
    self.patch('/users/%(user_id)s' % {'user_id': self.user['id']}, body={'user': user}, token=CONF.admin_token, expected_status=http.client.BAD_REQUEST)