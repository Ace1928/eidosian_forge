from boto.dynamodb.batch import BatchList
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb import exceptions as dynamodb_exceptions
import time
def update_from_response(self, response):
    """
        Update the state of the Table object based on the response
        data received from Amazon DynamoDB.
        """
    if 'Table' in response:
        self._dict.update(response['Table'])
    elif 'TableDescription' in response:
        self._dict.update(response['TableDescription'])
    if 'KeySchema' in self._dict:
        self._schema = Schema(self._dict['KeySchema'])