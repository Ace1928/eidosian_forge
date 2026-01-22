from boto.ec2.ec2object import TaggedEC2Object
from boto.ec2.launchspecification import LaunchSpecification
class SpotInstanceStatus(object):
    """
    Contains the status of a Spot Instance Request.

    :ivar code: Status code of the request.
    :ivar message: The description for the status code for the Spot request.
    :ivar update_time: Time the status was stated.
    """

    def __init__(self, code=None, update_time=None, message=None):
        self.code = code
        self.update_time = update_time
        self.message = message

    def __repr__(self):
        return '<Status: %s>' % self.code

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'code':
            self.code = value
        elif name == 'message':
            self.message = value
        elif name == 'updateTime':
            self.update_time = value