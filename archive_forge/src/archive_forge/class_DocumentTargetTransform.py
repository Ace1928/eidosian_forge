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
class DocumentTargetTransform(SphinxPostTransform):
    """Add :doc label to the first section of each document."""
    default_priority = 400
    formats = ('latex',)

    def run(self, **kwargs: Any) -> None:
        for node in self.document.findall(addnodes.start_of_file):
            section = node.next_node(nodes.section)
            if section:
                section['ids'].append(':doc')