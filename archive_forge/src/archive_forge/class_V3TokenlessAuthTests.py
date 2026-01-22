import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
class V3TokenlessAuthTests(utils.TestCase):

    def setUp(self):
        super(V3TokenlessAuthTests, self).setUp()
        self.auth_url = uuid.uuid4().hex

    def create(self, **kwargs):
        kwargs.setdefault('auth_url', self.auth_url)
        loader = loading.get_plugin_loader('v3tokenlessauth')
        return loader.load_from_options(**kwargs)

    def test_basic(self):
        domain_id = uuid.uuid4().hex
        domain_name = uuid.uuid4().hex
        project_id = uuid.uuid4().hex
        project_name = uuid.uuid4().hex
        project_domain_id = uuid.uuid4().hex
        project_domain_name = uuid.uuid4().hex
        tla = self.create(domain_id=domain_id, domain_name=domain_name, project_id=project_id, project_name=project_name, project_domain_id=project_domain_id, project_domain_name=project_domain_name)
        self.assertEqual(domain_id, tla.domain_id)
        self.assertEqual(domain_name, tla.domain_name)
        self.assertEqual(project_id, tla.project_id)
        self.assertEqual(project_name, tla.project_name)
        self.assertEqual(project_domain_id, tla.project_domain_id)
        self.assertEqual(project_domain_name, tla.project_domain_name)

    def test_missing_parameters(self):
        self.assertRaises(exceptions.OptionError, self.create, domain_id=None)
        self.assertRaises(exceptions.OptionError, self.create, domain_name=None)
        self.assertRaises(exceptions.OptionError, self.create, project_id=None)
        self.assertRaises(exceptions.OptionError, self.create, project_name=None)
        self.assertRaises(exceptions.OptionError, self.create, project_domain_id=None)
        self.assertRaises(exceptions.OptionError, self.create, project_domain_name=None)
        self.assertRaises(exceptions.OptionError, self.create, project_domain_id=uuid.uuid4().hex)
        self.assertRaises(exceptions.OptionError, self.create, project_domain_name=uuid.uuid4().hex)
        self.assertRaises(exceptions.OptionError, self.create, project_name=uuid.uuid4().hex)