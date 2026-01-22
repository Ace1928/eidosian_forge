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
class ASTDeclaration(ASTBaseBase):

    def __init__(self, objectType: str, directiveType: str, declaration: Union[DeclarationType, ASTFunctionParameter], semicolon: bool=False) -> None:
        self.objectType = objectType
        self.directiveType = directiveType
        self.declaration = declaration
        self.semicolon = semicolon
        self.symbol: Symbol = None
        self.enumeratorScopedSymbol: Symbol = None

    def clone(self) -> 'ASTDeclaration':
        return ASTDeclaration(self.objectType, self.directiveType, self.declaration.clone(), self.semicolon)

    @property
    def name(self) -> ASTNestedName:
        decl = cast(DeclarationType, self.declaration)
        return decl.name

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        if self.objectType != 'function':
            return None
        decl = cast(ASTType, self.declaration)
        return decl.function_params

    def get_id(self, version: int, prefixed: bool=True) -> str:
        if self.objectType == 'enumerator' and self.enumeratorScopedSymbol:
            return self.enumeratorScopedSymbol.declaration.get_id(version, prefixed)
        id_ = self.declaration.get_id(version, self.objectType, self.symbol)
        if prefixed:
            return _id_prefix[version] + id_
        else:
            return id_

    def get_newest_id(self) -> str:
        return self.get_id(_max_id, True)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = transform(self.declaration)
        if self.semicolon:
            res += ';'
        return res

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', options: Dict) -> None:
        verify_description_mode(mode)
        assert self.symbol
        signode['is_multiline'] = True
        mainDeclNode = addnodes.desc_signature_line()
        mainDeclNode.sphinx_line_type = 'declarator'
        mainDeclNode['add_permalink'] = not self.symbol.isRedeclaration
        signode += mainDeclNode
        if self.objectType == 'member':
            pass
        elif self.objectType == 'function':
            pass
        elif self.objectType == 'macro':
            pass
        elif self.objectType == 'struct':
            mainDeclNode += addnodes.desc_sig_keyword('struct', 'struct')
            mainDeclNode += addnodes.desc_sig_space()
        elif self.objectType == 'union':
            mainDeclNode += addnodes.desc_sig_keyword('union', 'union')
            mainDeclNode += addnodes.desc_sig_space()
        elif self.objectType == 'enum':
            mainDeclNode += addnodes.desc_sig_keyword('enum', 'enum')
            mainDeclNode += addnodes.desc_sig_space()
        elif self.objectType == 'enumerator':
            mainDeclNode += addnodes.desc_sig_keyword('enumerator', 'enumerator')
            mainDeclNode += addnodes.desc_sig_space()
        elif self.objectType == 'type':
            decl = cast(ASTType, self.declaration)
            prefix = decl.get_type_declaration_prefix()
            mainDeclNode += addnodes.desc_sig_keyword(prefix, prefix)
            mainDeclNode += addnodes.desc_sig_space()
        else:
            raise AssertionError()
        self.declaration.describe_signature(mainDeclNode, mode, env, self.symbol)
        if self.semicolon:
            mainDeclNode += addnodes.desc_sig_punctuation(';', ';')