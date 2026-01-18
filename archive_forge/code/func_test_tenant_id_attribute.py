import sqlalchemy as sa
from neutron_lib import context
from neutron_lib.db import model_base
from neutron_lib.tests.unit.db import _base as db_base
def test_tenant_id_attribute(self):
    foo = TestTable(tenant_id='tenant')
    self.assertEqual('tenant', foo.project_id)
    self.assertEqual('tenant', foo.tenant_id)