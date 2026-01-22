from boto.resultset import ResultSet
from boto.ec2.ec2object import EC2Object
from boto.utils import parse_ts
class InstanceCount(object):

    def __init__(self, connection=None, state=None, instance_count=None):
        self.state = state
        self.instance_count = instance_count

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'state':
            self.state = value
        elif name == 'instanceCount':
            self.instance_count = int(value)
        else:
            setattr(self, name, value)