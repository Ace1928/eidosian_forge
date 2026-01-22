from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_sig_element(nodes.inline, _desc_classes_injector):
    """Common parent class of nodes for inline text of a signature."""
    classes: List[str] = []

    def __init__(self, rawsource: str='', text: str='', *children: Element, **attributes: Any) -> None:
        super().__init__(rawsource, text, *children, **attributes)
        self['classes'].extend(self.classes)