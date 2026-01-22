from time import time
from typing import Optional
from zope.interface import Interface, implementer
from twisted.protocols import pcp
class ShapedProtocolFactory:
    """
    Dispense C{Protocols} with traffic shaping on their transports.

    Usage::

        myserver = SomeFactory()
        myserver.protocol = ShapedProtocolFactory(myserver.protocol,
                                                  bucketFilter)

    Where C{SomeServerFactory} is a L{twisted.internet.protocol.Factory}, and
    C{bucketFilter} is an instance of L{HierarchicalBucketFilter}.
    """

    def __init__(self, protoClass, bucketFilter):
        """
        Tell me what to wrap and where to get buckets.

        @param protoClass: The class of C{Protocol} this will generate
          wrapped instances of.
        @type protoClass: L{Protocol<twisted.internet.interfaces.IProtocol>}
          class
        @param bucketFilter: The filter which will determine how
          traffic is shaped.
        @type bucketFilter: L{HierarchicalBucketFilter}.
        """
        self.protocol = protoClass
        self.bucketFilter = bucketFilter

    def __call__(self, *a, **kw):
        """
        Make a C{Protocol} instance with a shaped transport.

        Any parameters will be passed on to the protocol's initializer.

        @returns: A C{Protocol} instance with a L{ShapedTransport}.
        """
        proto = self.protocol(*a, **kw)
        origMakeConnection = proto.makeConnection

        def makeConnection(transport):
            bucket = self.bucketFilter.getBucketFor(transport)
            shapedTransport = ShapedTransport(transport, bucket)
            return origMakeConnection(shapedTransport)
        proto.makeConnection = makeConnection
        return proto