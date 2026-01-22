import mock # Yes, we only test the rolling backport
import testtools
from fixtures import (
class TestMockMultiple(testtools.TestCase):

    def test_mock_multiple_with_replacement(self):
        self.useFixture(MockPatchMultiple('%s.Foo' % __name__, bar=mocking_bar))
        instance = Foo()
        self.assertEqual(instance.bar(), 'mocked!')

    def test_mock_patch_without_replacement(self):
        self.useFixture(MockPatchMultiple('%s.Foo' % __name__, bar=MockPatchMultiple.DEFAULT))
        instance = Foo()
        self.assertIsInstance(instance.bar(), mock.MagicMock)