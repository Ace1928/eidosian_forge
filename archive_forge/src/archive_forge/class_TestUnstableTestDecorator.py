from neutron_lib.tests import _base as base
from neutron_lib.utils import test
class TestUnstableTestDecorator(base.BaseTestCase):

    @test.unstable_test('some bug')
    def test_unstable_pass(self):
        self.assertIsNone(None)

    @test.unstable_test('some other bug')
    def test_unstable_fail(self):
        self.assertIsNotNone(None)