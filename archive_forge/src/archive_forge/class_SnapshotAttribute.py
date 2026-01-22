from boto.ec2.ec2object import TaggedEC2Object
from boto.ec2.zone import Zone
class SnapshotAttribute(object):

    def __init__(self, parent=None):
        self.snapshot_id = None
        self.attrs = {}

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'createVolumePermission':
            self.name = 'create_volume_permission'
        elif name == 'group':
            if 'groups' in self.attrs:
                self.attrs['groups'].append(value)
            else:
                self.attrs['groups'] = [value]
        elif name == 'userId':
            if 'user_ids' in self.attrs:
                self.attrs['user_ids'].append(value)
            else:
                self.attrs['user_ids'] = [value]
        elif name == 'snapshotId':
            self.snapshot_id = value
        else:
            setattr(self, name, value)