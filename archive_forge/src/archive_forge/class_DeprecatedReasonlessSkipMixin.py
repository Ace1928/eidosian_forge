from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class DeprecatedReasonlessSkipMixin:

    def test_1(self):
        raise SkipTest()