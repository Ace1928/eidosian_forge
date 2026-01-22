import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
def _test_bad_variables(type_, input_):
    result = schema.execute(f'query Test($input: {type_}){{ {type_.lower()}(in: $input) }}', variables={'input': input_})
    assert isinstance(result.errors, list)
    assert len(result.errors) == 1
    assert isinstance(result.errors[0], GraphQLError)
    assert result.data is None