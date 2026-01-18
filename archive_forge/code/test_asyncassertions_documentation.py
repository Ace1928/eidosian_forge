import unittest as pyunit
from twisted.internet import defer
from twisted.python import failure
from twisted.trial import unittest

        A single wrong assertFailure should fail the whole test.
        