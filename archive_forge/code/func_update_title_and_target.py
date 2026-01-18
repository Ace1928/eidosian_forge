import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type
import docutils.parsers.rst.directives
import docutils.parsers.rst.roles
import docutils.parsers.rst.states
from docutils import nodes, utils
from docutils.nodes import Element, Node, TextElement, system_message
from sphinx import addnodes
from sphinx.locale import _, __
from sphinx.util import ws_re
from sphinx.util.docutils import ReferenceRole, SphinxRole
from sphinx.util.typing import RoleFunction
def update_title_and_target(self, title: str, target: str) -> Tuple[str, str]:
    if not self.has_explicit_title:
        if title.endswith('()'):
            title = title[:-2]
        if self.config.add_function_parentheses:
            title += '()'
    if target.endswith('()'):
        target = target[:-2]
    return (title, target)