import uuid
from keystone.common import provider_api
from keystone.common import sql
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit.assignment import test_core
from keystone.tests.unit.backend import core_sql
class SqlRoleModels(core_sql.BaseBackendSqlModels):

    def test_role_model(self):
        cols = (('id', sql.String, 64), ('name', sql.String, 255), ('domain_id', sql.String, 64))
        self.assertExpectedSchema('role', cols)