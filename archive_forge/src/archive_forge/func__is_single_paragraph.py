from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union, cast
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst.states import Inliner
from sphinx import addnodes
from sphinx.environment import BuildEnvironment
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.typing import TextlikeNode
def _is_single_paragraph(node: nodes.field_body) -> bool:
    """True if the node only contains one paragraph (and system messages)."""
    if len(node) == 0:
        return False
    elif len(node) > 1:
        for subnode in node[1:]:
            if not isinstance(subnode, nodes.system_message):
                return False
    if isinstance(node[0], nodes.paragraph):
        return True
    return False