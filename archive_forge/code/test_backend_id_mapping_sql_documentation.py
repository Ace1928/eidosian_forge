import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
from keystone.identity.mapping_backends import mapping
from keystone.tests import unit
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit import test_backend_sql
Set of tests for checking SQL Identity ID Mapping.