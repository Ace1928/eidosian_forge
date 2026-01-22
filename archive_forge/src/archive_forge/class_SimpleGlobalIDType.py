from graphql_relay import from_global_id, to_global_id
from ..types import ID, UUID
from ..types.base import BaseType
from typing import Type
class SimpleGlobalIDType(BaseGlobalIDType):
    """
    Simple global ID type: simply the id of the object.
    To be used carefully as the user is responsible for ensuring that the IDs are indeed global
    (otherwise it could cause request caching issues).
    """
    graphene_type = ID

    @classmethod
    def resolve_global_id(cls, info, global_id):
        _type = info.return_type.graphene_type._meta.name
        return (_type, global_id)

    @classmethod
    def to_global_id(cls, _type, _id):
        return _id