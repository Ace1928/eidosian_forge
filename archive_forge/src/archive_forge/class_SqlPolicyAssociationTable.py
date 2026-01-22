from keystone.common import sql
from keystone.tests.unit import test_backend_endpoint_policy
from keystone.tests.unit import test_backend_sql
class SqlPolicyAssociationTable(test_backend_sql.SqlModels):
    """Set of tests for checking SQL Policy Association Mapping."""

    def test_policy_association_mapping(self):
        cols = (('id', sql.String, 64), ('policy_id', sql.String, 64), ('endpoint_id', sql.String, 64), ('service_id', sql.String, 64), ('region_id', sql.String, 64))
        self.assertExpectedSchema('policy_association', cols)