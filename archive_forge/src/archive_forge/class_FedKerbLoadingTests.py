from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit import utils as test_utils
class FedKerbLoadingTests(test_utils.TestCase):

    def test_options(self):
        opts = [o.name for o in loading.get_plugin_loader('v3fedkerb').get_options()]
        allowed_opts = ['system-scope', 'domain-id', 'domain-name', 'identity-provider', 'project-id', 'project-name', 'project-domain-id', 'project-domain-name', 'protocol', 'trust-id', 'auth-url', 'mutual-auth']
        self.assertCountEqual(allowed_opts, opts)

    def create(self, **kwargs):
        loader = loading.get_plugin_loader('v3fedkerb')
        return loader.load_from_options(**kwargs)

    def test_load_none(self):
        self.assertRaises(exceptions.MissingRequiredOptions, self.create)

    def test_load(self):
        self.create(auth_url='auth_url', identity_provider='idp', protocol='protocol')