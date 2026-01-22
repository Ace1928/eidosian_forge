from boto.ec2.securitygroup import SecurityGroup
class EC2SecurityGroup(object):
    """
    Describes an EC2 security group for use in a DBSecurityGroup
    """

    def __init__(self, parent=None):
        self.parent = parent
        self.name = None
        self.owner_id = None

    def __repr__(self):
        return 'EC2SecurityGroup:%s' % self.name

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'EC2SecurityGroupName':
            self.name = value
        elif name == 'EC2SecurityGroupOwnerId':
            self.owner_id = value
        else:
            setattr(self, name, value)