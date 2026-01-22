from boto.ec2.ec2object import TaggedEC2Object
class CustomerGateway(TaggedEC2Object):

    def __init__(self, connection=None):
        super(CustomerGateway, self).__init__(connection)
        self.id = None
        self.type = None
        self.state = None
        self.ip_address = None
        self.bgp_asn = None

    def __repr__(self):
        return 'CustomerGateway:%s' % self.id

    def endElement(self, name, value, connection):
        if name == 'customerGatewayId':
            self.id = value
        elif name == 'ipAddress':
            self.ip_address = value
        elif name == 'type':
            self.type = value
        elif name == 'state':
            self.state = value
        elif name == 'bgpAsn':
            self.bgp_asn = int(value)
        else:
            setattr(self, name, value)