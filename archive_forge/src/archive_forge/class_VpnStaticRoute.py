import boto
from datetime import datetime
from boto.resultset import ResultSet
from boto.ec2.ec2object import TaggedEC2Object
class VpnStaticRoute(object):
    """
    Represents a static route for a VPN connection.

    :ivar destination_cidr_block: The CIDR block associated with the local
        subnet of the customer data center.
    :ivar source: Indicates how the routes were provided.
    :ivar state: The current state of the static route.
    """

    def __init__(self, destination_cidr_block=None, source=None, state=None):
        self.destination_cidr_block = destination_cidr_block
        self.source = source
        self.available = state

    def __repr__(self):
        return 'VpnStaticRoute: %s' % self.destination_cidr_block

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'destinationCidrBlock':
            self.destination_cidr_block = value
        elif name == 'source':
            self.source = value
        elif name == 'state':
            self.state = value
        else:
            setattr(self, name, value)