from boto.ec2.ec2object import TaggedEC2Object
from boto.resultset import ResultSet
class NetworkAclEntry(object):

    def __init__(self, connection=None):
        self.rule_number = None
        self.protocol = None
        self.rule_action = None
        self.egress = None
        self.cidr_block = None
        self.port_range = PortRange()
        self.icmp = Icmp()

    def __repr__(self):
        return 'Acl:%s' % self.rule_number

    def startElement(self, name, attrs, connection):
        if name == 'portRange':
            return self.port_range
        elif name == 'icmpTypeCode':
            return self.icmp
        else:
            return None

    def endElement(self, name, value, connection):
        if name == 'cidrBlock':
            self.cidr_block = value
        elif name == 'egress':
            self.egress = value
        elif name == 'protocol':
            self.protocol = value
        elif name == 'ruleAction':
            self.rule_action = value
        elif name == 'ruleNumber':
            self.rule_number = value