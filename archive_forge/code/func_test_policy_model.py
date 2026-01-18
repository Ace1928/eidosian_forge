from keystone.common import sql
from keystone.policy.backends import sql as sql_driver
from keystone.tests import unit
from keystone.tests.unit.backend import core_sql
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.policy.backends import test_base
def test_policy_model(self):
    cols = (('id', sql.String, 64), ('blob', sql.JsonBlob, None), ('type', sql.String, 255), ('extra', sql.JsonBlob, None))
    self.assertExpectedSchema('policy', cols)