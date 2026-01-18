from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
def test_proxy_wrapped_membership(self):
    proxy_factory = proxy.ImageMembershipFactory(self.factory, proxy_class=FakeProxy, proxy_kwargs={'a': 1})
    self.factory.result = 'tyrion'
    membership = proxy_factory.new_image_member('jaime', 'cersei')
    self.assertIsInstance(membership, FakeProxy)
    self.assertEqual('tyrion', membership.base)
    self.assertEqual({'a': 1}, membership.kwargs)
    self.assertEqual('jaime', self.factory.image)
    self.assertEqual('cersei', self.factory.member_id)