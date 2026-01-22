from graphql_relay import from_global_id, to_global_id
from ..types import ID, UUID
from ..types.base import BaseType
from typing import Type
class BaseGlobalIDType:
    """
    Base class that define the required attributes/method for a type.
    """
    graphene_type = ID

    @classmethod
    def resolve_global_id(cls, info, global_id):
        raise NotImplementedError

    @classmethod
    def to_global_id(cls, _type, _id):
        raise NotImplementedError