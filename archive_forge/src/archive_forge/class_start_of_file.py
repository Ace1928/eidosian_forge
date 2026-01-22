from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class start_of_file(nodes.Element):
    """Node to mark start of a new file, used in the LaTeX builder only."""