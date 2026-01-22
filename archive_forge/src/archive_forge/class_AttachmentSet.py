from boto.resultset import ResultSet
from boto.ec2.tag import Tag
from boto.ec2.ec2object import TaggedEC2Object
class AttachmentSet(object):
    """
    Represents an EBS attachmentset.

    :ivar id: The unique ID of the volume.
    :ivar instance_id: The unique ID of the attached instance
    :ivar status: The status of the attachment
    :ivar attach_time: Attached since
    :ivar device: The device the instance has mapped
    """

    def __init__(self):
        self.id = None
        self.instance_id = None
        self.status = None
        self.attach_time = None
        self.device = None

    def __repr__(self):
        return 'AttachmentSet:%s' % self.id

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'volumeId':
            self.id = value
        elif name == 'instanceId':
            self.instance_id = value
        elif name == 'status':
            self.status = value
        elif name == 'attachTime':
            self.attach_time = value
        elif name == 'device':
            self.device = value
        else:
            setattr(self, name, value)