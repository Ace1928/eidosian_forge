from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_sig_keyword_type(desc_sig_element):
    """Node for a keyword which is a built-in type in a signature."""
    classes = ['kt']