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
def warn_missing_reference(self, refdoc: str, typ: str, target: str, node: pending_xref, domain: Optional[Domain]) -> None:
    warn = node.get('refwarn')
    if self.config.nitpicky:
        warn = True
        dtype = '%s:%s' % (domain.name, typ) if domain else typ
        if self.config.nitpick_ignore:
            if (dtype, target) in self.config.nitpick_ignore:
                warn = False
            if (not domain or domain.name == 'std') and (typ, target) in self.config.nitpick_ignore:
                warn = False
        if self.config.nitpick_ignore_regex:

            def matches_ignore(entry_type: str, entry_target: str) -> bool:
                for ignore_type, ignore_target in self.config.nitpick_ignore_regex:
                    if re.fullmatch(ignore_type, entry_type) and re.fullmatch(ignore_target, entry_target):
                        return True
                return False
            if matches_ignore(dtype, target):
                warn = False
            if (not domain or domain.name == 'std') and matches_ignore(typ, target):
                warn = False
    if not warn:
        return
    if self.app.emit_firstresult('warn-missing-reference', domain, node):
        return
    elif domain and typ in domain.dangling_warnings:
        msg = domain.dangling_warnings[typ] % {'target': target}
    elif node.get('refdomain', 'std') not in ('', 'std'):
        msg = __('%s:%s reference target not found: %s') % (node['refdomain'], typ, target)
    else:
        msg = __('%r reference target not found: %s') % (typ, target)
    logger.warning(msg, location=node, type='ref', subtype=typ)