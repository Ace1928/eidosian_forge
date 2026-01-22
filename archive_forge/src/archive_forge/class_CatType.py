import re
from pytest import raises
from graphql import parse, get_introspection_query, validate
from ...types import Schema, ObjectType, Interface
from ...types import String, Int, List, Field
from ..depth_limit import depth_limit_validator
class CatType(ObjectType):

    class meta:
        name = 'Cat'
        interfaces = (PetType,)