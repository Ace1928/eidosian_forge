from unittest import skipIf
from twisted.trial.unittest import TestCase
@skipIf(True, '')
def test_shouldNeverRunWithEmptyReason(self) -> None:
    raise Exception('Test should skip and never reach here')