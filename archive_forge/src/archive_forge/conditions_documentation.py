from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
Builds the condition expression and the dictionary of placeholders.

        :type condition: ConditionBase
        :param condition: A condition to be built into a condition expression
            string with any necessary placeholders.

        :type is_key_condition: Boolean
        :param is_key_condition: True if the expression is for a
            KeyConditionExpression. False otherwise.

        :rtype: (string, dict, dict)
        :returns: Will return a string representing the condition with
            placeholders inserted where necessary, a dictionary of
            placeholders for attribute names, and a dictionary of
            placeholders for attribute values. Here is a sample return value:

            ('#n0 = :v0', {'#n0': 'myattribute'}, {':v1': 'myvalue'})
        