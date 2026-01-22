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
class FootnoteCollector(nodes.NodeVisitor):
    """Collect footnotes and footnote references on the document"""

    def __init__(self, document: nodes.document) -> None:
        self.auto_footnotes: List[nodes.footnote] = []
        self.used_footnote_numbers: Set[str] = set()
        self.footnote_refs: List[nodes.footnote_reference] = []
        super().__init__(document)

    def unknown_visit(self, node: Node) -> None:
        pass

    def unknown_departure(self, node: Node) -> None:
        pass

    def visit_footnote(self, node: nodes.footnote) -> None:
        if node.get('auto'):
            self.auto_footnotes.append(node)
        else:
            for name in node['names']:
                self.used_footnote_numbers.add(name)

    def visit_footnote_reference(self, node: nodes.footnote_reference) -> None:
        self.footnote_refs.append(node)