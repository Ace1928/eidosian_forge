from boto.ec2.ec2object import TaggedEC2Object
from boto.resultset import ResultSet
class InternetGateway(TaggedEC2Object):

    def __init__(self, connection=None):
        super(InternetGateway, self).__init__(connection)
        self.id = None
        self.attachments = []

    def __repr__(self):
        return 'InternetGateway:%s' % self.id

    def startElement(self, name, attrs, connection):
        result = super(InternetGateway, self).startElement(name, attrs, connection)
        if result is not None:
            return result
        if name == 'attachmentSet':
            self.attachments = ResultSet([('item', InternetGatewayAttachment)])
            return self.attachments
        else:
            return None

    def endElement(self, name, value, connection):
        if name == 'internetGatewayId':
            self.id = value
        else:
            setattr(self, name, value)