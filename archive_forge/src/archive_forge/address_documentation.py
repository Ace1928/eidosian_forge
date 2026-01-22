from zope.interface import implementer
from twisted.internet.interfaces import IAddress
from twisted.python import util

    Object representing an SSH Transport endpoint.

    This is used to ensure that any code inspecting this address and
    attempting to construct a similar connection based upon it is not
    mislead into creating a transport which is not similar to the one it is
    indicating.

    @ivar address: An instance of an object which implements I{IAddress} to
        which this transport address is connected.
    