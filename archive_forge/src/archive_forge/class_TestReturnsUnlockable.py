from testtools.matchers import *
from . import CapturedCall, TestCase, TestCaseWithTransport
from .matchers import *
class TestReturnsUnlockable(TestCase):

    def test___str__(self):
        matcher = ReturnsUnlockable(StubTree(True))
        self.assertEqual('ReturnsUnlockable(lockable_thing=I am da tree)', str(matcher))

    def test_match(self):
        stub_tree = StubTree(False)
        matcher = ReturnsUnlockable(stub_tree)
        self.assertThat(matcher.match(lambda: FakeUnlockable()), Equals(None))

    def test_mismatch(self):
        stub_tree = StubTree(True)
        matcher = ReturnsUnlockable(stub_tree)
        mismatch = matcher.match(lambda: FakeUnlockable())
        self.assertNotEqual(None, mismatch)
        self.assertThat(mismatch.describe(), Equals('I am da tree is locked'))