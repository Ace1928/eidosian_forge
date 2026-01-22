import re
import unicodedata
from typing import (TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Set, Tuple, Type,
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import Directive
from docutils.parsers.rst.states import Inliner
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.locale import __
from sphinx.util import logging
def apply_source_workaround(node: Element) -> None:
    if isinstance(node, nodes.classifier) and (not node.rawsource):
        logger.debug('[i18n] PATCH: %r to have source, line and rawsource: %s', get_full_module_name(node), repr_domxml(node))
        definition_list_item = node.parent
        node.source = definition_list_item.source
        node.line = definition_list_item.line - 1
        node.rawsource = node.astext()
    elif isinstance(node, nodes.classifier) and (not node.source):
        node.source = node.parent.source
    if isinstance(node, nodes.image) and node.source is None:
        logger.debug('[i18n] PATCH: %r to have source, line: %s', get_full_module_name(node), repr_domxml(node))
        node.source, node.line = (node.parent.source, node.parent.line)
    if isinstance(node, nodes.title) and node.source is None:
        logger.debug('[i18n] PATCH: %r to have source: %s', get_full_module_name(node), repr_domxml(node))
        node.source, node.line = (node.parent.source, node.parent.line)
    if isinstance(node, nodes.term):
        logger.debug('[i18n] PATCH: %r to have rawsource: %s', get_full_module_name(node), repr_domxml(node))
        for classifier in reversed(list(node.parent.findall(nodes.classifier))):
            node.rawsource = re.sub('\\s*:\\s*%s' % re.escape(classifier.astext()), '', node.rawsource)
    if isinstance(node, nodes.topic) and node.source is None:
        logger.debug('[i18n] PATCH: %r to have source, line: %s', get_full_module_name(node), repr_domxml(node))
        node.source, node.line = (node.parent.source, node.parent.line)
    if isinstance(node, nodes.literal_block) and node.source is None:
        node.source = get_node_source(node)
    if not node.rawsource:
        node.rawsource = node.astext()
    if node.source and node.rawsource:
        return
    if isinstance(node, (nodes.rubric, nodes.line, nodes.image, nodes.field_name)):
        logger.debug('[i18n] PATCH: %r to have source and line: %s', get_full_module_name(node), repr_domxml(node))
        node.source = get_node_source(node) or ''
        node.line = 0
        return