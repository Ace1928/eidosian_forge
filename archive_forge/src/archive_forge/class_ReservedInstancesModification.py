from boto.resultset import ResultSet
from boto.ec2.ec2object import EC2Object
from boto.utils import parse_ts
class ReservedInstancesModification(object):

    def __init__(self, connection=None, modification_id=None, reserved_instances=None, modification_results=None, create_date=None, update_date=None, effective_date=None, status=None, status_message=None, client_token=None):
        self.connection = connection
        self.modification_id = modification_id
        self.reserved_instances = reserved_instances
        self.modification_results = modification_results
        self.create_date = create_date
        self.update_date = update_date
        self.effective_date = effective_date
        self.status = status
        self.status_message = status_message
        self.client_token = client_token

    def startElement(self, name, attrs, connection):
        if name == 'reservedInstancesSet':
            self.reserved_instances = ResultSet([('item', ReservedInstance)])
            return self.reserved_instances
        elif name == 'modificationResultSet':
            self.modification_results = ResultSet([('item', ModificationResult)])
            return self.modification_results
        return None

    def endElement(self, name, value, connection):
        if name == 'reservedInstancesModificationId':
            self.modification_id = value
        elif name == 'createDate':
            self.create_date = parse_ts(value)
        elif name == 'updateDate':
            self.update_date = parse_ts(value)
        elif name == 'effectiveDate':
            self.effective_date = parse_ts(value)
        elif name == 'status':
            self.status = value
        elif name == 'statusMessage':
            self.status_message = value
        elif name == 'clientToken':
            self.client_token = value
        else:
            setattr(self, name, value)