from typing import Any, cast
from ....error import GraphQLError
from ....language import ArgumentNode, EnumValueNode, FieldNode, ObjectFieldNode
from ....type import GraphQLInputObjectType, get_named_type, is_input_object_type
from .. import ValidationRule
No deprecated

    A GraphQL document is only valid if all selected fields and all used enum values
    have not been deprecated.

    Note: This rule is optional and is not part of the Validation section of the GraphQL
    Specification. The main purpose of this rule is detection of deprecated usages and
    not necessarily to forbid their use when querying a service.
    