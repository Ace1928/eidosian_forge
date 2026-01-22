from unittest import skipIf
from twisted.trial.unittest import TestCase
class SkipDecoratorUsedOnMethods(TestCase):
    """
    Only methods where @skipIf decorator is used should be skipped.
    """

    @skipIf(True, 'skipIf decorator used so skip test')
    def test_shouldNeverRun(self) -> None:
        raise Exception('Test should skip and never reach here')

    @skipIf(True, '')
    def test_shouldNeverRunWithEmptyReason(self) -> None:
        raise Exception('Test should skip and never reach here')

    def test_shouldShouldRun(self) -> None:
        self.assertTrue(True, 'Test should run and not be skipped')

    @skipIf(False, 'should not skip')
    def test_shouldShouldRunWithSkipIfFalse(self) -> None:
        self.assertTrue(True, 'Test should run and not be skipped')

    @skipIf(False, '')
    def test_shouldShouldRunWithSkipIfFalseEmptyReason(self) -> None:
        self.assertTrue(True, 'Test should run and not be skipped')