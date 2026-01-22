from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
class TestProxyRepoWrapping(test_utils.BaseTestCase):

    def setUp(self):
        super(TestProxyRepoWrapping, self).setUp()
        self.fake_repo = FakeRepo()
        self.proxy_repo = proxy.Repo(self.fake_repo, item_proxy_class=FakeProxy, item_proxy_kwargs={'a': 1})

    def _test_method(self, name, base_result, *args, **kwargs):
        self.fake_repo.result = base_result
        method = getattr(self.proxy_repo, name)
        proxy_result = method(*args, **kwargs)
        self.assertIsInstance(proxy_result, FakeProxy)
        self.assertEqual(base_result, proxy_result.base)
        self.assertEqual(0, len(proxy_result.args))
        self.assertEqual({'a': 1}, proxy_result.kwargs)
        self.assertEqual(args, self.fake_repo.args)
        self.assertEqual(kwargs, self.fake_repo.kwargs)

    def test_get(self):
        self.fake_repo.result = 'snarf'
        result = self.proxy_repo.get('some-id')
        self.assertIsInstance(result, FakeProxy)
        self.assertEqual(('some-id',), self.fake_repo.args)
        self.assertEqual({}, self.fake_repo.kwargs)
        self.assertEqual('snarf', result.base)
        self.assertEqual(tuple(), result.args)
        self.assertEqual({'a': 1}, result.kwargs)

    def test_list(self):
        self.fake_repo.result = ['scratch', 'sniff']
        results = self.proxy_repo.list(2, prefix='s')
        self.assertEqual((2,), self.fake_repo.args)
        self.assertEqual({'prefix': 's'}, self.fake_repo.kwargs)
        self.assertEqual(2, len(results))
        for i in range(2):
            self.assertIsInstance(results[i], FakeProxy)
            self.assertEqual(self.fake_repo.result[i], results[i].base)
            self.assertEqual(tuple(), results[i].args)
            self.assertEqual({'a': 1}, results[i].kwargs)

    def _test_method_with_proxied_argument(self, name, result, **kwargs):
        self.fake_repo.result = result
        item = FakeProxy('snoop')
        method = getattr(self.proxy_repo, name)
        proxy_result = method(item)
        self.assertEqual(('snoop',), self.fake_repo.args)
        self.assertEqual(kwargs, self.fake_repo.kwargs)
        if result is None:
            self.assertIsNone(proxy_result)
        else:
            self.assertIsInstance(proxy_result, FakeProxy)
            self.assertEqual(result, proxy_result.base)
            self.assertEqual(tuple(), proxy_result.args)
            self.assertEqual({'a': 1}, proxy_result.kwargs)

    def test_add(self):
        self._test_method_with_proxied_argument('add', 'dog')

    def test_add_with_no_result(self):
        self._test_method_with_proxied_argument('add', None)

    def test_save(self):
        self._test_method_with_proxied_argument('save', 'dog', from_state=None)

    def test_save_with_no_result(self):
        self._test_method_with_proxied_argument('save', None, from_state=None)

    def test_remove(self):
        self._test_method_with_proxied_argument('remove', 'dog')

    def test_remove_with_no_result(self):
        self._test_method_with_proxied_argument('remove', None)