import sqlalchemy as sa
from neutron_lib import context
from neutron_lib.db import model_base
from neutron_lib.tests.unit.db import _base as db_base
def test_project_id_attribute(self):
    foo = TestTable(project_id='project')
    self.assertEqual('project', foo.project_id)
    self.assertEqual('project', foo.tenant_id)