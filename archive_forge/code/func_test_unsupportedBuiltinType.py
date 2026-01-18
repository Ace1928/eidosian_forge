import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
def test_unsupportedBuiltinType(self):
    """
        Banana does not support arbitrary builtin types like L{type}.
        L{banana.Banana.sendEncoded} raises L{banana.BananaError} if called
        with an instance of L{type}.
        """
    self._unsupportedTypeTest(type, 'builtins.type')