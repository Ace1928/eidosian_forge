from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
class TestImageFactory(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImageFactory, self).setUp()
        self.factory = FakeImageFactory()

    def test_proxy_plain(self):
        proxy_factory = proxy.ImageFactory(self.factory)
        self.factory.result = 'eddard'
        image = proxy_factory.new_image(a=1, b='two')
        self.assertEqual('eddard', image)
        self.assertEqual({'a': 1, 'b': 'two'}, self.factory.kwargs)

    def test_proxy_wrapping(self):
        proxy_factory = proxy.ImageFactory(self.factory, proxy_class=FakeProxy, proxy_kwargs={'dog': 'bark'})
        self.factory.result = 'stark'
        image = proxy_factory.new_image(a=1, b='two')
        self.assertIsInstance(image, FakeProxy)
        self.assertEqual('stark', image.base)
        self.assertEqual({'a': 1, 'b': 'two'}, self.factory.kwargs)