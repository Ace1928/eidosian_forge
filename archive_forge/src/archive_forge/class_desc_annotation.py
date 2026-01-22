from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_annotation(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    """Node for signature annotations (not Python 3-style annotations)."""