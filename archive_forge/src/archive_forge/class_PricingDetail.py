from boto.resultset import ResultSet
from boto.ec2.ec2object import EC2Object
from boto.utils import parse_ts
class PricingDetail(object):

    def __init__(self, connection=None, price=None, count=None):
        self.price = price
        self.count = count

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        setattr(self, name, value)