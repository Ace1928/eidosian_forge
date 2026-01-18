from ..objecttype import ObjectType
from ..schema import Schema
from ..uuid import UUID
from ..structures import NonNull
def test_uuidstring_query_variable():
    uuid_value = 'dfeb3bcf-70fd-11e7-a61a-6003088f8204'
    result = schema.execute('query Test($uuid: UUID){ uuid(input: $uuid) }', variables={'uuid': uuid_value})
    assert not result.errors
    assert result.data == {'uuid': uuid_value}