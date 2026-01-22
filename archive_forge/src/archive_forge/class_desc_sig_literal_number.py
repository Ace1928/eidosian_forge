from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class desc_sig_literal_number(desc_sig_element):
    """Node for a numeric literal in a signature."""
    classes = ['m']