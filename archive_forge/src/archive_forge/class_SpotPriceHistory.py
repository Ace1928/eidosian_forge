from boto.ec2.ec2object import EC2Object
class SpotPriceHistory(EC2Object):

    def __init__(self, connection=None):
        super(SpotPriceHistory, self).__init__(connection)
        self.price = 0.0
        self.instance_type = None
        self.product_description = None
        self.timestamp = None
        self.availability_zone = None

    def __repr__(self):
        return 'SpotPriceHistory(%s):%2f' % (self.instance_type, self.price)

    def endElement(self, name, value, connection):
        if name == 'instanceType':
            self.instance_type = value
        elif name == 'spotPrice':
            self.price = float(value)
        elif name == 'productDescription':
            self.product_description = value
        elif name == 'timestamp':
            self.timestamp = value
        elif name == 'availabilityZone':
            self.availability_zone = value
        else:
            setattr(self, name, value)