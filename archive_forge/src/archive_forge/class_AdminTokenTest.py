from testtools import matchers
from keystoneauth1.loading._plugins import admin_token as loader
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
class AdminTokenTest(utils.TestCase):

    def test_token_endpoint_options(self):
        opt_names = [opt.name for opt in loader.AdminToken().get_options()]
        self.assertThat(opt_names, matchers.HasLength(2))
        self.assertIn('token', opt_names)
        self.assertIn('endpoint', opt_names)

    def test_token_endpoint_deprecated_options(self):
        endpoint_opt = [opt for opt in loader.AdminToken().get_options() if opt.name == 'endpoint'][0]
        opt_names = [opt.name for opt in endpoint_opt.deprecated]
        self.assertEqual(['url'], opt_names)