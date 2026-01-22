import re
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, TypeVar,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.roles import SphinxRole, XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (ASTAttributeList, ASTBaseBase, ASTBaseParenExprList,
from sphinx.util.docfields import Field, GroupedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx.util.typing import OptionSpec
class ASTClass(ASTBase):

    def __init__(self, name: ASTNestedName, final: bool, bases: List[ASTBaseClass], attrs: ASTAttributeList) -> None:
        self.name = name
        self.final = final
        self.bases = bases
        self.attrs = attrs

    def get_id(self, version: int, objectType: str, symbol: 'Symbol') -> str:
        return symbol.get_full_nested_name().get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.attrs))
        if len(self.attrs) != 0:
            res.append(' ')
        res.append(transform(self.name))
        if self.final:
            res.append(' final')
        if len(self.bases) > 0:
            res.append(' : ')
            first = True
            for b in self.bases:
                if not first:
                    res.append(', ')
                first = False
                res.append(transform(b))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        verify_description_mode(mode)
        self.attrs.describe_signature(signode)
        if len(self.attrs) != 0:
            signode += addnodes.desc_sig_space()
        self.name.describe_signature(signode, mode, env, symbol=symbol)
        if self.final:
            signode += addnodes.desc_sig_space()
            signode += addnodes.desc_sig_keyword('final', 'final')
        if len(self.bases) > 0:
            signode += addnodes.desc_sig_space()
            signode += addnodes.desc_sig_punctuation(':', ':')
            signode += addnodes.desc_sig_space()
            for b in self.bases:
                b.describe_signature(signode, mode, env, symbol=symbol)
                signode += addnodes.desc_sig_punctuation(',', ',')
                signode += addnodes.desc_sig_space()
            signode.pop()
            signode.pop()