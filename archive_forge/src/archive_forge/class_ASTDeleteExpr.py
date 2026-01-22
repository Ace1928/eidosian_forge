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
class ASTDeleteExpr(ASTExpression):

    def __init__(self, rooted: bool, array: bool, expr: ASTExpression):
        self.rooted = rooted
        self.array = array
        self.expr = expr

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.rooted:
            res.append('::')
        res.append('delete ')
        if self.array:
            res.append('[] ')
        res.append(transform(self.expr))
        return ''.join(res)

    def get_id(self, version: int) -> str:
        if self.array:
            id = 'da'
        else:
            id = 'dl'
        return id + self.expr.get_id(version)

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        if self.rooted:
            signode += addnodes.desc_sig_punctuation('::', '::')
        signode += addnodes.desc_sig_keyword('delete', 'delete')
        signode += addnodes.desc_sig_space()
        if self.array:
            signode += addnodes.desc_sig_punctuation('[]', '[]')
            signode += addnodes.desc_sig_space()
        self.expr.describe_signature(signode, mode, env, symbol)