from typing import Any, Dict, Optional, cast
from ...error import GraphQLError
from ...language import (
from ...pyutils import Undefined
from ...type import GraphQLNonNull, GraphQLSchema, GraphQLType, is_non_null_type
from ...utilities import type_from_ast, is_type_sub_type_of
from . import ValidationContext, ValidationRule
class VariablesInAllowedPositionRule(ValidationRule):
    """Variables in allowed position

    Variable usages must be compatible with the arguments they are passed to.

    See https://spec.graphql.org/draft/#sec-All-Variable-Usages-are-Allowed
    """

    def __init__(self, context: ValidationContext):
        super().__init__(context)
        self.var_def_map: Dict[str, Any] = {}

    def enter_operation_definition(self, *_args: Any) -> None:
        self.var_def_map.clear()

    def leave_operation_definition(self, operation: OperationDefinitionNode, *_args: Any) -> None:
        var_def_map = self.var_def_map
        usages = self.context.get_recursive_variable_usages(operation)
        for usage in usages:
            node, type_ = (usage.node, usage.type)
            default_value = usage.default_value
            var_name = node.name.value
            var_def = var_def_map.get(var_name)
            if var_def and type_:
                schema = self.context.schema
                var_type = type_from_ast(schema, var_def.type)
                if var_type and (not allowed_variable_usage(schema, var_type, var_def.default_value, type_, default_value)):
                    self.report_error(GraphQLError(f"Variable '${var_name}' of type '{var_type}' used in position expecting type '{type_}'.", [var_def, node]))

    def enter_variable_definition(self, node: VariableDefinitionNode, *_args: Any) -> None:
        self.var_def_map[node.variable.name.value] = node