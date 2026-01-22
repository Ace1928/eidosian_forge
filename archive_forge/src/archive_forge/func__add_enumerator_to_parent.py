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
def _add_enumerator_to_parent(self, ast: ASTDeclaration) -> None:
    assert ast.objectType == 'enumerator'
    symbol = ast.symbol
    assert symbol
    assert symbol.ident is not None
    parentSymbol = symbol.parent
    assert parentSymbol
    if parentSymbol.parent is None:
        return
    parentDecl = parentSymbol.declaration
    if parentDecl is None:
        return
    if parentDecl.objectType != 'enum':
        return
    if parentDecl.directiveType != 'enum':
        return
    targetSymbol = parentSymbol.parent
    s = targetSymbol.find_identifier(symbol.ident, matchSelf=False, recurseInAnon=True, searchInSiblings=False)
    if s is not None:
        return
    declClone = symbol.declaration.clone()
    declClone.enumeratorScopedSymbol = symbol
    Symbol(parent=targetSymbol, ident=symbol.ident, declaration=declClone, docname=self.env.docname, line=self.get_source_info()[1])