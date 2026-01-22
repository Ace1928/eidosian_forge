from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_returns(desc_type):
    """Node for a "returns" annotation (a la -> in Python)."""

    def astext(self) -> str:
        return ' -> ' + super().astext()