import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
from oslo_db import exception as db_exception
from oslo_db import options
from oslo_log import log
import sqlalchemy
from sqlalchemy import exc
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import sql
from keystone.common.sql import core
import keystone.conf
from keystone.credential.providers import fernet as credential_provider
from keystone import exception
from keystone.identity.backends import sql_model as identity_sql
from keystone.resource.backends import base as resource
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.limit import test_backends as limit_tests
from keystone.tests.unit.policy import test_backends as policy_tests
from keystone.tests.unit.resource import test_backends as resource_tests
from keystone.tests.unit.trust import test_backends as trust_tests
from keystone.trust.backends import sql as trust_sql
def test_update_project_returns_extra(self):
    """Test for backward compatibility with an essex/folsom bug.

        Non-indexed attributes were returned in an 'extra' attribute, instead
        of on the entity itself; for consistency and backwards compatibility,
        those attributes should be included twice.

        This behavior is specific to the SQL driver.

        """
    arbitrary_key = uuid.uuid4().hex
    arbitrary_value = uuid.uuid4().hex
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    project[arbitrary_key] = arbitrary_value
    ref = PROVIDERS.resource_api.create_project(project['id'], project)
    self.assertEqual(arbitrary_value, ref[arbitrary_key])
    self.assertNotIn('extra', ref)
    ref['name'] = uuid.uuid4().hex
    ref = PROVIDERS.resource_api.update_project(ref['id'], ref)
    self.assertEqual(arbitrary_value, ref[arbitrary_key])
    self.assertEqual(arbitrary_value, ref['extra'][arbitrary_key])