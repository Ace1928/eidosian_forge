import re
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, TypeVar,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.roles import SphinxRole, XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (ASTAttributeList, ASTBaseBase, ASTBaseParenExprList,
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx.util.typing import OptionSpec
class ASTDeclSpecsSimple(ASTBaseBase):

    def __init__(self, storage: str, threadLocal: str, inline: bool, restrict: bool, volatile: bool, const: bool, attrs: ASTAttributeList) -> None:
        self.storage = storage
        self.threadLocal = threadLocal
        self.inline = inline
        self.restrict = restrict
        self.volatile = volatile
        self.const = const
        self.attrs = attrs

    def mergeWith(self, other: 'ASTDeclSpecsSimple') -> 'ASTDeclSpecsSimple':
        if not other:
            return self
        return ASTDeclSpecsSimple(self.storage or other.storage, self.threadLocal or other.threadLocal, self.inline or other.inline, self.volatile or other.volatile, self.const or other.const, self.restrict or other.restrict, self.attrs + other.attrs)

    def _stringify(self, transform: StringifyTransform) -> str:
        res: List[str] = []
        if len(self.attrs) != 0:
            res.append(transform(self.attrs))
        if self.storage:
            res.append(self.storage)
        if self.threadLocal:
            res.append(self.threadLocal)
        if self.inline:
            res.append('inline')
        if self.restrict:
            res.append('restrict')
        if self.volatile:
            res.append('volatile')
        if self.const:
            res.append('const')
        return ' '.join(res)

    def describe_signature(self, modifiers: List[Node]) -> None:

        def _add(modifiers: List[Node], text: str) -> None:
            if len(modifiers) != 0:
                modifiers.append(addnodes.desc_sig_space())
            modifiers.append(addnodes.desc_sig_keyword(text, text))
        if len(modifiers) != 0 and len(self.attrs) != 0:
            modifiers.append(addnodes.desc_sig_space())
        tempNode = nodes.TextElement()
        self.attrs.describe_signature(tempNode)
        modifiers.extend(tempNode.children)
        if self.storage:
            _add(modifiers, self.storage)
        if self.threadLocal:
            _add(modifiers, self.threadLocal)
        if self.inline:
            _add(modifiers, 'inline')
        if self.restrict:
            _add(modifiers, 'restrict')
        if self.volatile:
            _add(modifiers, 'volatile')
        if self.const:
            _add(modifiers, 'const')