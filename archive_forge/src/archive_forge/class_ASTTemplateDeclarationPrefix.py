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
class ASTTemplateDeclarationPrefix(ASTBase):

    def __init__(self, templates: List[Union[ASTTemplateParams, ASTTemplateIntroduction]]) -> None:
        self.templates = templates

    def get_requires_clause_in_last(self) -> Optional['ASTRequiresClause']:
        if self.templates is None:
            return None
        lastList = self.templates[-1]
        if not isinstance(lastList, ASTTemplateParams):
            return None
        return lastList.requiresClause

    def get_id_except_requires_clause_in_last(self, version: int) -> str:
        assert version >= 2
        res = []
        lastIndex = len(self.templates) - 1
        for i, t in enumerate(self.templates):
            if isinstance(t, ASTTemplateParams):
                res.append(t.get_id(version, excludeRequires=i == lastIndex))
            else:
                res.append(t.get_id(version))
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        for t in self.templates:
            res.append(transform(t))
        return ''.join(res)

    def describe_signature(self, signode: desc_signature, mode: str, env: 'BuildEnvironment', symbol: 'Symbol', lineSpec: bool) -> None:
        verify_description_mode(mode)
        for t in self.templates:
            t.describe_signature_as_introducer(signode, 'lastIsName', env, symbol, lineSpec)