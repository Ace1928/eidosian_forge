from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
class ExpectedFailure(SynchronousTestCase):
    """
    Hold a test that has an expected failure with an exception that has a
    large string representation.
    """

    def test_expectedFailureGreaterThan64k(self) -> None:
        """
        Fail, but expectedly.
        """
        raise RuntimeError('x' * (2 ** 16 + 1))
    test_expectedFailureGreaterThan64k.todo = 'short todo string'