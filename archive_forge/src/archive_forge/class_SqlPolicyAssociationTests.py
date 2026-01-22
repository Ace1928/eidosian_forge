from keystone.common import sql
from keystone.tests.unit import test_backend_endpoint_policy
from keystone.tests.unit import test_backend_sql
class SqlPolicyAssociationTests(test_backend_sql.SqlTests, test_backend_endpoint_policy.PolicyAssociationTests):

    def load_fixtures(self, fixtures):
        super(SqlPolicyAssociationTests, self).load_fixtures(fixtures)
        self.load_sample_data()