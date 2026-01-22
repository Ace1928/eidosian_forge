import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, cast
from docutils import nodes
from docutils.nodes import Element, Node
from sphinx import addnodes
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.domains import Domain
from sphinx.errors import NoUri
from sphinx.locale import __
from sphinx.transforms import SphinxTransform
from sphinx.util import logging
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.nodes import find_pending_xref_condition, process_only_nodes
class SigElementFallbackTransform(SphinxPostTransform):
    """Fallback various desc_* nodes to inline if translator does not support them."""
    default_priority = 200

    def run(self, **kwargs: Any) -> None:

        def has_visitor(translator: Type[nodes.NodeVisitor], node: Type[Element]) -> bool:
            return hasattr(translator, 'visit_%s' % node.__name__)
        translator = self.app.builder.get_translator_class()
        if isinstance(translator, SphinxTranslator):
            return
        if not all((has_visitor(translator, node) for node in addnodes.SIG_ELEMENTS)):
            self.fallback(addnodes.desc_sig_element)
        if not has_visitor(translator, addnodes.desc_inline):
            self.fallback(addnodes.desc_inline)

    def fallback(self, nodeType: Any) -> None:
        for node in self.document.findall(nodeType):
            newnode = nodes.inline()
            newnode.update_all_atts(node)
            newnode.extend(node)
            node.replace_self(newnode)