from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class number_reference(nodes.reference):
    """Node for number references, similar to pending_xref."""