import sqlalchemy as sa
from neutron_lib import context
from neutron_lib.db import model_base
from neutron_lib.tests.unit.db import _base as db_base
class TestModelBase(db_base.SqlTestCase):

    def setUp(self):
        super(TestModelBase, self).setUp()
        self.ctx = context.Context('user', 'project')
        self.session = self.ctx.session

    def test_model_base(self):
        foo = TestTable(name='meh')
        self.assertEqual('meh', foo.name)
        self.assertIn('meh', str(foo))
        cols = [k for k, _v in foo]
        self.assertIn('name', cols)

    def test_get_set_tenant_id_tenant(self):
        foo = TestTable(tenant_id='tenant')
        self.assertEqual('tenant', foo.get_tenant_id())
        foo.set_tenant_id('project')
        self.assertEqual('project', foo.get_tenant_id())

    def test_get_set_tenant_id_project(self):
        foo = TestTable(project_id='project')
        self.assertEqual('project', foo.get_tenant_id())
        foo.set_tenant_id('tenant')
        self.assertEqual('tenant', foo.get_tenant_id())

    def test_project_id_attribute(self):
        foo = TestTable(project_id='project')
        self.assertEqual('project', foo.project_id)
        self.assertEqual('project', foo.tenant_id)

    def test_tenant_id_attribute(self):
        foo = TestTable(tenant_id='tenant')
        self.assertEqual('tenant', foo.project_id)
        self.assertEqual('tenant', foo.tenant_id)