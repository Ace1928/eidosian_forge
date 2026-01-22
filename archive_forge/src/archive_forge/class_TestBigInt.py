from ..objecttype import ObjectType, Field
from ..scalars import Scalar, Int, BigInt, Float, String, Boolean
from ..schema import Schema
from graphql import Undefined
from graphql.language.ast import IntValueNode
class TestBigInt:

    def test_query(self):
        """
        Test that a normal query works.
        """
        value = 2 ** 31
        result = schema.execute('{ optional { bigInt(input: %s) } }' % value)
        assert not result.errors
        assert result.data == {'optional': {'bigInt': value}}

    def test_optional_input(self):
        """
        Test that we can provide a null value to an optional input
        """
        result = schema.execute('{ optional { bigInt(input: null) } }')
        assert not result.errors
        assert result.data == {'optional': {'bigInt': None}}

    def test_invalid_input(self):
        """
        Test that if an invalid type is provided we get an error
        """
        result = schema.execute('{ optional { bigInt(input: "20") } }')
        assert result.errors
        assert len(result.errors) == 1
        assert result.errors[0].message == 'Expected value of type \'BigInt\', found "20".'
        result = schema.execute('{ optional { bigInt(input: "a") } }')
        assert result.errors
        assert len(result.errors) == 1
        assert result.errors[0].message == 'Expected value of type \'BigInt\', found "a".'
        result = schema.execute('{ optional { bigInt(input: true) } }')
        assert result.errors
        assert len(result.errors) == 1
        assert result.errors[0].message == "Expected value of type 'BigInt', found true."