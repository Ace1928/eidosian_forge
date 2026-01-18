from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
def test_proxy_wrapped_image(self):
    proxy_factory = proxy.ImageMembershipFactory(self.factory, proxy_class=FakeProxy)
    self.factory.result = 'tyrion'
    image = FakeProxy('jaime')
    membership = proxy_factory.new_image_member(image, 'cersei')
    self.assertIsInstance(membership, FakeProxy)
    self.assertIsInstance(self.factory.image, FakeProxy)
    self.assertEqual('cersei', self.factory.member_id)