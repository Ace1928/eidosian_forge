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
class PropagateDescDomain(SphinxPostTransform):
    """Add the domain name of the parent node as a class in each desc_signature node."""
    default_priority = 200

    def run(self, **kwargs: Any) -> None:
        for node in self.document.findall(addnodes.desc_signature):
            if node.parent.get('domain'):
                node['classes'].append(node.parent['domain'])