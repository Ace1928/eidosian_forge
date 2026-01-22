import mistralclient.auth.keystone
from mistralclient.tests.unit.v2 import base
class TestKeystone(base.BaseClientV2Test):

    def setUp(self):
        super(TestKeystone, self).setUp()
        self.keystone = mistralclient.auth.keystone.KeystoneAuthHandler()

    def test_get_auth_token(self):
        auth = self.keystone._get_auth(auth_token='token', auth_url='url', project_id='project_id')
        self.assertEqual('url', auth.auth_url)
        elements = auth.get_cache_id_elements()
        self.assertIsNotNone(elements['token'])
        self.assertIsNotNone(elements['project_id'])

    def test_remove_domain(self):
        params = {'param1': 'p', 'target_param2': 'p2', 'user_domain_param3': 'p3', 'target_project_domain_param4': 'p4'}
        dedomained = self.keystone._remove_domain(params)
        self.assertIn('param1', dedomained)
        self.assertIn('target_param2', dedomained)
        self.assertNotIn('user_domain_param3', dedomained)
        self.assertNotIn('target_project_domain_param4', dedomained)

    def test_separate_target_reqs(self):
        params = {'a': 1, 'target_b': 2, 'c': 3, 'target_d': 4, 'target_target': 5, 'param_target': 6}
        nontarget, target = self.keystone._separate_target_reqs(params)
        self.assertIn('a', nontarget)
        self.assertIn('c', nontarget)
        self.assertIn('param_target', nontarget)
        self.assertIn('b', target)
        self.assertIn('d', target)
        self.assertIn('target', target)

    def test_verify(self):
        self.assertTrue(self.keystone._verification_needed('', False))
        self.assertFalse(self.keystone._verification_needed('', True))
        self.assertFalse(self.keystone._verification_needed('cert', True))
        self.assertEqual(self.keystone._verification_needed('cert', False), 'cert')