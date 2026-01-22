from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
class TranslatedError(Exception):
    """
    Translated exception type when testing an exception translation.
    """