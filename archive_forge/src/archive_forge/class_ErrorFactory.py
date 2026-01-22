from typing import Dict, List, Tuple
from twisted.internet.testing import StringTransport
from twisted.protocols import postfix
from twisted.trial import unittest
class ErrorFactory:
    """
            Factory that raises an error on key lookup.
            """

    def get(self, key):
        raise Exception('This is a test error')