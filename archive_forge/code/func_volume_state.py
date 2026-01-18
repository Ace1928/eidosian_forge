from boto.resultset import ResultSet
from boto.ec2.tag import Tag
from boto.ec2.ec2object import TaggedEC2Object
def volume_state(self):
    """
        Returns the state of the volume.  Same value as the status attribute.
        """
    return self.status