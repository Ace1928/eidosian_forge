from datetime import datetime
from boto.resultset import ResultSet
class StackResourceSummary(object):

    def __init__(self, connection=None):
        self.connection = connection
        self.last_updated_time = None
        self.logical_resource_id = None
        self.physical_resource_id = None
        self.resource_status = None
        self.resource_status_reason = None
        self.resource_type = None

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'LastUpdatedTime':
            try:
                self.last_updated_time = datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')
            except ValueError:
                self.last_updated_time = datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%fZ')
        elif name == 'LogicalResourceId':
            self.logical_resource_id = value
        elif name == 'PhysicalResourceId':
            self.physical_resource_id = value
        elif name == 'ResourceStatus':
            self.resource_status = value
        elif name == 'ResourceStatusReason':
            self.resource_status_reason = value
        elif name == 'ResourceType':
            self.resource_type = value
        else:
            setattr(self, name, value)

    def __repr__(self):
        return 'StackResourceSummary:%s (%s)' % (self.logical_resource_id, self.resource_type)