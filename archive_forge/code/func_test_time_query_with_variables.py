import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
def test_time_query_with_variables(sample_time):
    isoformat = sample_time.isoformat()
    result = schema.execute('\n        query GetTime($time: Time) {\n          literal: time(at: "%s")\n          value: time(at: $time)\n        }\n        ' % isoformat, variable_values={'time': isoformat})
    assert not result.errors
    assert result.data == {'literal': isoformat, 'value': isoformat}