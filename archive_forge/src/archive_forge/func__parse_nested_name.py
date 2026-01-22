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
def _parse_nested_name(self) -> ASTNestedName:
    names: List[Any] = []
    self.skip_ws()
    rooted = False
    if self.skip_string('.'):
        rooted = True
    while 1:
        self.skip_ws()
        if not self.match(identifier_re):
            self.fail('Expected identifier in nested name.')
        identifier = self.matched_text
        if identifier in _keywords:
            self.fail('Expected identifier in nested name, got keyword: %s' % identifier)
        if self.matched_text in self.config.c_extra_keywords:
            msg = 'Expected identifier, got user-defined keyword: %s.' + ' Remove it from c_extra_keywords to allow it as identifier.\n' + 'Currently c_extra_keywords is %s.'
            self.fail(msg % (self.matched_text, str(self.config.c_extra_keywords)))
        ident = ASTIdentifier(identifier)
        names.append(ident)
        self.skip_ws()
        if not self.skip_string('.'):
            break
    return ASTNestedName(names, rooted)