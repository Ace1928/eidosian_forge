from twisted.internet import address
from twisted.trial import unittest
from .. import _v1parser
from .._exceptions import InvalidNetworkProtocol, InvalidProxyHeader, MissingAddressData

        Test that parsing leaves overflow bytes in the buffer.
        