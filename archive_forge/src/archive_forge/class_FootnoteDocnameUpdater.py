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
class FootnoteDocnameUpdater(SphinxTransform):
    """Add docname to footnote and footnote_reference nodes."""
    default_priority = 700
    TARGET_NODES = (nodes.footnote, nodes.footnote_reference)

    def apply(self, **kwargs: Any) -> None:
        matcher = NodeMatcher(*self.TARGET_NODES)
        for node in self.document.findall(matcher):
            node['docname'] = self.env.docname