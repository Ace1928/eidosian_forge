from unittest import skipIf
from twisted.trial.unittest import TestCase
@skipIf(True, 'skipIf decorator used so skip test')
def test_shouldNeverRun(self) -> None:
    raise Exception('Test should skip and never reach here')