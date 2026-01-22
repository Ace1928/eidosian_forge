import boto
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.types import (NonBooleanDynamizer, Dynamizer, FILTER_OPERATORS,
from boto.exception import JSONResponseError
def _build_filters(self, filter_kwargs, using=QUERY_OPERATORS):
    """
        An internal method for taking query/scan-style ``**kwargs`` & turning
        them into the raw structure DynamoDB expects for filtering.
        """
    if filter_kwargs is None:
        return
    filters = {}
    for field_and_op, value in filter_kwargs.items():
        field_bits = field_and_op.split('__')
        fieldname = '__'.join(field_bits[:-1])
        try:
            op = using[field_bits[-1]]
        except KeyError:
            raise exceptions.UnknownFilterTypeError("Operator '%s' from '%s' is not recognized." % (field_bits[-1], field_and_op))
        lookup = {'AttributeValueList': [], 'ComparisonOperator': op}
        if field_bits[-1] == 'null':
            del lookup['AttributeValueList']
            if value is False:
                lookup['ComparisonOperator'] = 'NOT_NULL'
            else:
                lookup['ComparisonOperator'] = 'NULL'
        elif field_bits[-1] == 'between':
            if len(value) == 2 and isinstance(value, (list, tuple)):
                lookup['AttributeValueList'].append(self._dynamizer.encode(value[0]))
                lookup['AttributeValueList'].append(self._dynamizer.encode(value[1]))
        elif field_bits[-1] == 'in':
            for val in value:
                lookup['AttributeValueList'].append(self._dynamizer.encode(val))
        else:
            if isinstance(value, (list, tuple)):
                value = set(value)
            lookup['AttributeValueList'].append(self._dynamizer.encode(value))
        filters[fieldname] = lookup
    return filters