from boto.resultset import ResultSet
from boto.ec2.ec2object import EC2Object
from boto.utils import parse_ts
class ReservedInstanceListing(EC2Object):

    def __init__(self, connection=None, listing_id=None, id=None, create_date=None, update_date=None, status=None, status_message=None, client_token=None):
        self.connection = connection
        self.listing_id = listing_id
        self.id = id
        self.create_date = create_date
        self.update_date = update_date
        self.status = status
        self.status_message = status_message
        self.client_token = client_token

    def startElement(self, name, attrs, connection):
        if name == 'instanceCounts':
            self.instance_counts = ResultSet([('item', InstanceCount)])
            return self.instance_counts
        elif name == 'priceSchedules':
            self.price_schedules = ResultSet([('item', PriceSchedule)])
            return self.price_schedules
        return None

    def endElement(self, name, value, connection):
        if name == 'reservedInstancesListingId':
            self.listing_id = value
        elif name == 'reservedInstancesId':
            self.id = value
        elif name == 'createDate':
            self.create_date = value
        elif name == 'updateDate':
            self.update_date = value
        elif name == 'status':
            self.status = value
        elif name == 'statusMessage':
            self.status_message = value
        else:
            setattr(self, name, value)