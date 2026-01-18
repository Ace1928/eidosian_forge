from ..objecttype import ObjectType, Field
from ..scalars import Scalar, Int, BigInt, Float, String, Boolean
from ..schema import Schema
from graphql import Undefined
from graphql.language.ast import IntValueNode
def test_optional_input(self):
    """
        Test that we can provide a null value to an optional input
        """
    result = schema.execute('{ optional { string(input: null) } }')
    assert not result.errors
    assert result.data == {'optional': {'string': None}}