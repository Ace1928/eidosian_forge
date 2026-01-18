import re
import sys
from copy import copy
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional,
from docutils import nodes
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.locale import _, __
from sphinx.roles import EmphasizedLiteral, XRefRole
from sphinx.util import docname_join, logging, ws_re
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import clean_astext, make_id, make_refnode
from sphinx.util.typing import OptionSpec, RoleFunction
def token_xrefs(text: str, productionGroup: str='') -> List[Node]:
    if len(productionGroup) != 0:
        productionGroup += ':'
    retnodes: List[Node] = []
    pos = 0
    for m in token_re.finditer(text):
        if m.start() > pos:
            txt = text[pos:m.start()]
            retnodes.append(nodes.Text(txt))
        token = m.group(1)
        if ':' in token:
            if token[0] == '~':
                _, title = token.split(':')
                target = token[1:]
            elif token[0] == ':':
                title = token[1:]
                target = title
            else:
                title = token
                target = token
        else:
            title = token
            target = productionGroup + token
        refnode = pending_xref(title, reftype='token', refdomain='std', reftarget=target)
        refnode += nodes.literal(token, title, classes=['xref'])
        retnodes.append(refnode)
        pos = m.end()
    if pos < len(text):
        retnodes.append(nodes.Text(text[pos:]))
    return retnodes