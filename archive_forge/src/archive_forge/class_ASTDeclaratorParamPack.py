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
class ASTDeclaratorParamPack(ASTDeclarator):

    def __init__(self, next: ASTDeclarator) -> None:
        assert next
        self.next = next

    @property
    def name(self) -> ASTNestedName:
        return self.next.name

    @name.setter
    def name(self, name: ASTNestedName) -> None:
        self.next.name = name

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.next.function_params

    @property
    def trailingReturn(self) -> 'ASTType':
        return self.next.trailingReturn

    @property
    def isPack(self) -> bool:
        return True

    def require_space_after_declSpecs(self) -> bool:
        return False

    def _stringify(self, transform: StringifyTransform) -> str:
        res = transform(self.next)
        if self.next.name:
            res = ' ' + res
        return '...' + res

    def get_modifiers_id(self, version: int) -> str:
        return self.next.get_modifiers_id(version)

    def get_param_id(self, version: int) -> str:
        return self.next.get_param_id(version)

    def get_ptr_suffix_id(self, version: int) -> str:
        if version == 1:
            return 'Dp' + self.next.get_ptr_suffix_id(version)
        else:
            return self.next.get_ptr_suffix_id(version) + 'Dp'

    def get_type_id(self, version: int, returnTypeId: str) -> str:
        assert version >= 2
        return self.next.get_type_id(version, returnTypeId='Dp' + returnTypeId)

    def is_function_type(self) -> bool:
        return self.next.is_function_type()

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        verify_description_mode(mode)
        signode += addnodes.desc_sig_punctuation('...', '...')
        if self.next.name:
            signode += addnodes.desc_sig_space()
        self.next.describe_signature(signode, mode, env, symbol)