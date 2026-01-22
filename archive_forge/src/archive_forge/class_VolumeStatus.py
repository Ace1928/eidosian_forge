from boto.ec2.instancestatus import Status, Details
class VolumeStatus(object):
    """
    Represents an EC2 Volume status as reported by
    DescribeVolumeStatus request.

    :ivar id: The volume identifier.
    :ivar zone: The availability zone of the volume
    :ivar volume_status: A Status object that reports impaired
        functionality that arises from problems internal to the instance.
    :ivar events: A list of events relevant to the instance.
    :ivar actions: A list of events relevant to the instance.
    """

    def __init__(self, id=None, zone=None):
        self.id = id
        self.zone = zone
        self.volume_status = Status()
        self.events = None
        self.actions = None

    def __repr__(self):
        return 'VolumeStatus:%s' % self.id

    def startElement(self, name, attrs, connection):
        if name == 'eventsSet':
            self.events = EventSet()
            return self.events
        elif name == 'actionsSet':
            self.actions = ActionSet()
            return self.actions
        elif name == 'volumeStatus':
            return self.volume_status
        else:
            return None

    def endElement(self, name, value, connection):
        if name == 'volumeId':
            self.id = value
        elif name == 'availabilityZone':
            self.zone = value
        else:
            setattr(self, name, value)