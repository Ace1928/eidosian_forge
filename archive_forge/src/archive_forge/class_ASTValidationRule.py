from ...error import GraphQLError
from ...language.visitor import Visitor
from ..validation_context import (
class ASTValidationRule(Visitor):
    """Visitor for validation of an AST."""
    context: ASTValidationContext

    def __init__(self, context: ASTValidationContext):
        super().__init__()
        self.context = context

    def report_error(self, error: GraphQLError) -> None:
        self.context.report_error(error)