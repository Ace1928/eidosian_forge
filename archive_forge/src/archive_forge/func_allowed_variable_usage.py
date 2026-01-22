from typing import Any, Dict, Optional, cast
from ...error import GraphQLError
from ...language import (
from ...pyutils import Undefined
from ...type import GraphQLNonNull, GraphQLSchema, GraphQLType, is_non_null_type
from ...utilities import type_from_ast, is_type_sub_type_of
from . import ValidationContext, ValidationRule
def allowed_variable_usage(schema: GraphQLSchema, var_type: GraphQLType, var_default_value: Optional[ValueNode], location_type: GraphQLType, location_default_value: Any) -> bool:
    """Check for allowed variable usage.

    Returns True if the variable is allowed in the location it was found, which includes
    considering if default values exist for either the variable or the location at which
    it is located.
    """
    if is_non_null_type(location_type) and (not is_non_null_type(var_type)):
        has_non_null_variable_default_value = var_default_value is not None and (not isinstance(var_default_value, NullValueNode))
        has_location_default_value = location_default_value is not Undefined
        if not has_non_null_variable_default_value and (not has_location_default_value):
            return False
        location_type = cast(GraphQLNonNull, location_type)
        nullable_location_type = location_type.of_type
        return is_type_sub_type_of(schema, var_type, nullable_location_type)
    return is_type_sub_type_of(schema, var_type, location_type)