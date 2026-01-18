from typing import Collection, List, Optional, Type
from ..error import GraphQLError
from ..language import DocumentNode, ParallelVisitor, visit
from ..type import GraphQLSchema, assert_valid_schema
from ..pyutils import inspect, is_collection
from ..utilities import TypeInfo, TypeInfoVisitor
from .rules import ASTValidationRule
from .specified_rules import specified_rules, specified_sdl_rules
from .validation_context import SDLValidationContext, ValidationContext
def validate_sdl(document_ast: DocumentNode, schema_to_extend: Optional[GraphQLSchema]=None, rules: Optional[Collection[Type[ASTValidationRule]]]=None) -> List[GraphQLError]:
    """Validate an SDL document.

    For internal use only.
    """
    errors: List[GraphQLError] = []
    context = SDLValidationContext(document_ast, schema_to_extend, errors.append)
    if rules is None:
        rules = specified_sdl_rules
    visitors = [rule(context) for rule in rules]
    visit(document_ast, ParallelVisitor(visitors))
    return errors