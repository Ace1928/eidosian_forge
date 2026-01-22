from boto.ec2.ec2object import TaggedEC2Object
from boto.resultset import ResultSet
class NetworkAclAssociation(object):

    def __init__(self, connection=None):
        self.id = None
        self.subnet_id = None
        self.network_acl_id = None

    def __repr__(self):
        return 'NetworkAclAssociation:%s' % self.id

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'networkAclAssociationId':
            self.id = value
        elif name == 'networkAclId':
            self.network_acl_id = value
        elif name == 'subnetId':
            self.subnet_id = value