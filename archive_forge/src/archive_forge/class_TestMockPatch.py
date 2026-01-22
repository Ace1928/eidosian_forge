import mock # Yes, we only test the rolling backport
import testtools
from fixtures import (
class TestMockPatch(testtools.TestCase):

    def test_mock_patch_with_replacement(self):
        self.useFixture(MockPatch('%s.Foo.bar' % __name__, mocking_bar))
        instance = Foo()
        self.assertEqual(instance.bar(), 'mocked!')

    def test_mock_patch_without_replacement(self):
        self.useFixture(MockPatch('%s.Foo.bar' % __name__))
        instance = Foo()
        self.assertIsInstance(instance.bar(), mock.MagicMock)