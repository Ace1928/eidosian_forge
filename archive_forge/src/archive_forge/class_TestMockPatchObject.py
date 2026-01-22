import mock # Yes, we only test the rolling backport
import testtools
from fixtures import (
class TestMockPatchObject(testtools.TestCase):

    def test_mock_patch_object_with_replacement(self):
        self.useFixture(MockPatchObject(Foo, 'bar', mocking_bar))
        instance = Foo()
        self.assertEqual(instance.bar(), 'mocked!')

    def test_mock_patch_object_without_replacement(self):
        self.useFixture(MockPatchObject(Foo, 'bar'))
        instance = Foo()
        self.assertIsInstance(instance.bar(), mock.MagicMock)