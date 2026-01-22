from graphql_relay import from_global_id, to_global_id
from ..types import ID, UUID
from ..types.base import BaseType
from typing import Type
class DefaultGlobalIDType(BaseGlobalIDType):
    """
    Default global ID type: base64 encoded version of "<node type name>: <node id>".
    """
    graphene_type = ID

    @classmethod
    def resolve_global_id(cls, info, global_id):
        try:
            _type, _id = from_global_id(global_id)
            if not _type:
                raise ValueError('Invalid Global ID')
            return (_type, _id)
        except Exception as e:
            raise Exception(f'Unable to parse global ID "{global_id}". Make sure it is a base64 encoded string in the format: "TypeName:id". Exception message: {e}')

    @classmethod
    def to_global_id(cls, _type, _id):
        return to_global_id(_type, _id)