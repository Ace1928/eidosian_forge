import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase

        Check that a L{frozenset} can contain a circular reference and be
        serialized and unserialized without losing the reference.
        