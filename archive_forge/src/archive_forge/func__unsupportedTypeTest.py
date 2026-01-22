import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
def _unsupportedTypeTest(self, obj, name):
    """
        Assert that L{banana.Banana.sendEncoded} raises L{banana.BananaError}
        if called with the given object.

        @param obj: Some object that Banana does not support.
        @param name: The name of the type of the object.

        @raise: The failure exception is raised if L{Banana.sendEncoded} does
            not raise L{banana.BananaError} or if the message associated with the
            exception is not formatted to include the type of the unsupported
            object.
        """
    exc = self.assertRaises(banana.BananaError, self.enc.sendEncoded, obj)
    self.assertIn(f'Banana cannot send {name} objects', str(exc))