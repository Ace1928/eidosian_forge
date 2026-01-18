from keystone.common import sql
from keystone.endpoint_policy.backends import sql as sql_driver
from keystone.tests import unit
from keystone.tests.unit.backend import core_sql
from keystone.tests.unit.endpoint_policy.backends import test_base
from keystone.tests.unit.ksfixtures import database
def test_policy_association_model(self):
    cols = (('id', sql.String, 64), ('policy_id', sql.String, 64), ('endpoint_id', sql.String, 64), ('service_id', sql.String, 64), ('region_id', sql.String, 64))
    self.assertExpectedSchema('policy_association', cols)