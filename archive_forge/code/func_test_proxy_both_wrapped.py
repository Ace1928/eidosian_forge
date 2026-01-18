from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
def test_proxy_both_wrapped(self):

    class FakeProxy2(FakeProxy):
        pass
    proxy_factory = proxy.ImageMembershipFactory(self.factory, proxy_class=FakeProxy, proxy_kwargs={'b': 2})
    self.factory.result = 'tyrion'
    image = FakeProxy2('jaime')
    membership = proxy_factory.new_image_member(image, 'cersei')
    self.assertIsInstance(membership, FakeProxy)
    self.assertEqual('tyrion', membership.base)
    self.assertEqual({'b': 2}, membership.kwargs)
    self.assertIsInstance(self.factory.image, FakeProxy2)
    self.assertEqual('cersei', self.factory.member_id)