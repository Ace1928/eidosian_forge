from typing import Any, Dict, List, Optional, Set, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.transforms.references import Substitutions
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.latex.nodes import (captioned_literal_block, footnotemark, footnotetext,
from sphinx.domains.citation import CitationDomain
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.nodes import NodeMatcher
class LiteralBlockTransform(SphinxPostTransform):
    """Replace container nodes for literal_block by captioned_literal_block."""
    default_priority = 400
    formats = ('latex',)

    def run(self, **kwargs: Any) -> None:
        matcher = NodeMatcher(nodes.container, literal_block=True)
        for node in self.document.findall(matcher):
            newnode = captioned_literal_block('', *node.children, **node.attributes)
            node.replace_self(newnode)