from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_signature(_desc_classes_injector, nodes.Part, nodes.Inline, nodes.TextElement):
    """Node for a single object signature.

    As default the signature is a single-line signature.
    Set ``is_multiline = True`` to describe a multi-line signature.
    In that case all child nodes must be :py:class:`desc_signature_line` nodes.

    This node always has the classes ``sig``, ``sig-object``, and the domain it belongs to.
    """
    classes = ['sig', 'sig-object']

    @property
    def child_text_separator(self):
        if self.get('is_multiline'):
            return ' '
        else:
            return super().child_text_separator