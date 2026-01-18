from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure

            Make sure the result is more precise.
            On Python 3.11 or older this can be a float with ~ 0.00001
            in precision difference.
            See: https://github.com/python/cpython/issues/100425
            