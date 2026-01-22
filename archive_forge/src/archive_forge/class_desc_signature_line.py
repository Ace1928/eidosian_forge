from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_signature_line(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    """Node for a line in a multi-line object signature.

    It should only be used as a child of a :py:class:`desc_signature`
    with ``is_multiline`` set to ``True``.
    Set ``add_permalink = True`` for the line that should get the permalink.
    """
    sphinx_line_type = ''