import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
def test_time_query_variable(sample_time):
    isoformat = sample_time.isoformat()
    result = schema.execute('query Test($time: Time){ time(at: $time) }', variables={'time': sample_time})
    assert not result.errors
    assert result.data == {'time': isoformat}
    result = schema.execute('query Test($time: Time){ time(at: $time) }', variables={'time': isoformat})
    assert not result.errors
    assert result.data == {'time': isoformat}