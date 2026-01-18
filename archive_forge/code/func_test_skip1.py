from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
def test_skip1(self):
    raise SkipTest('skip1')