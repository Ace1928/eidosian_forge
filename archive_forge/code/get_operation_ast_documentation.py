from typing import Optional
from ..language import DocumentNode, OperationDefinitionNode
Get operation AST node.

    Returns an operation AST given a document AST and optionally an operation
    name. If a name is not provided, an operation is only returned if only one
    is provided in the document.
    