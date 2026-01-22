from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_optional(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    """Node for marking optional parts of the parameter list."""
    child_text_separator = ', '

    def astext(self) -> str:
        return '[' + super().astext() + ']'