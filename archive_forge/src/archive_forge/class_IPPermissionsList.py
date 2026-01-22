from boto.ec2.ec2object import TaggedEC2Object
from boto.exception import BotoClientError
class IPPermissionsList(list):

    def startElement(self, name, attrs, connection):
        if name == 'item':
            self.append(IPPermissions(self))
            return self[-1]
        return None

    def endElement(self, name, value, connection):
        pass