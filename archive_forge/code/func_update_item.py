import time
from binascii import crc32
import boto
from boto.connection import AWSAuthConnection
from boto.exception import DynamoDBResponseError
from boto.provider import Provider
from boto.dynamodb import exceptions as dynamodb_exceptions
from boto.compat import json
def update_item(self, table_name, key, attribute_updates, expected=None, return_values=None, object_hook=None):
    """
        Edits an existing item's attributes. You can perform a conditional
        update (insert a new attribute name-value pair if it doesn't exist,
        or replace an existing name-value pair if it has certain expected
        attribute values).

        :type table_name: str
        :param table_name: The name of the table.

        :type key: dict
        :param key: A Python version of the Key data structure
            defined by DynamoDB which identifies the item to be updated.

        :type attribute_updates: dict
        :param attribute_updates: A Python version of the AttributeUpdates
            data structure defined by DynamoDB.

        :type expected: dict
        :param expected: A Python version of the Expected
            data structure defined by DynamoDB.

        :type return_values: str
        :param return_values: Controls the return of attribute
            name-value pairs before then were changed.  Possible
            values are: None or 'ALL_OLD'. If 'ALL_OLD' is
            specified and the item is overwritten, the content
            of the old item is returned.
        """
    data = {'TableName': table_name, 'Key': key, 'AttributeUpdates': attribute_updates}
    if expected:
        data['Expected'] = expected
    if return_values:
        data['ReturnValues'] = return_values
    json_input = json.dumps(data)
    return self.make_request('UpdateItem', json_input, object_hook=object_hook)