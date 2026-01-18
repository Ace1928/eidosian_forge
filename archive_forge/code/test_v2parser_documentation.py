from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader

        Test that an initial payload of less than 16 bytes fails.
        