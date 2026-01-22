from boto.exception import BotoClientError
from boto.ec2.ec2object import TaggedEC2Object
from boto.resultset import ResultSet
from boto.ec2.group import Group
class PrivateIPAddress(object):

    def __init__(self, connection=None, private_ip_address=None, primary=None):
        self.connection = connection
        self.private_ip_address = private_ip_address
        self.primary = primary

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'privateIpAddress':
            self.private_ip_address = value
        elif name == 'primary':
            self.primary = True if value.lower() == 'true' else False

    def __repr__(self):
        return 'PrivateIPAddress(%s, primary=%s)' % (self.private_ip_address, self.primary)