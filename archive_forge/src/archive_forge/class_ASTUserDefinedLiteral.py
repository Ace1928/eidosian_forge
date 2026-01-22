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
class ASTUserDefinedLiteral(ASTLiteral):

    def __init__(self, literal: ASTLiteral, ident: ASTIdentifier):
        self.literal = literal
        self.ident = ident

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.literal) + transform(self.ident)

    def get_id(self, version: int) -> str:
        return 'clL_Zli{}E{}E'.format(self.ident.get_id(version), self.literal.get_id(version))

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        self.literal.describe_signature(signode, mode, env, symbol)
        self.ident.describe_signature(signode, 'udl', env, '', '', symbol)