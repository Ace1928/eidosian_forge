from keystone.common import sql
from keystone.tests.unit import test_backend_sql
def test_service_provider(self):
    cols = (('auth_url', sql.String, 256), ('id', sql.String, 64), ('enabled', sql.Boolean, None), ('description', sql.Text, None), ('relay_state_prefix', sql.String, 256), ('sp_url', sql.String, 256))
    self.assertExpectedSchema('service_provider', cols)