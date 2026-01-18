from boto.dynamodb.batch import BatchList
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb import exceptions as dynamodb_exceptions
import time
def update_throughput(self, read_units, write_units):
    """
        Update the ProvisionedThroughput for the Amazon DynamoDB Table.

        :type read_units: int
        :param read_units: The new value for ReadCapacityUnits.

        :type write_units: int
        :param write_units: The new value for WriteCapacityUnits.
        """
    self.layer2.update_throughput(self, read_units, write_units)