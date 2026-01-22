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
def _parse_expression_fallback(self, end: List[str], parser: Callable[[], ASTExpression], allow: bool=True) -> ASTExpression:
    prevPos = self.pos
    try:
        return parser()
    except DefinitionError as e:
        if not allow or not self.allowFallbackExpressionParsing:
            raise
        self.warn('Parsing of expression failed. Using fallback parser. Error was:\n%s' % e)
        self.pos = prevPos
    assert end is not None
    self.skip_ws()
    startPos = self.pos
    if self.match(_string_re):
        value = self.matched_text
    else:
        brackets = {'(': ')', '{': '}', '[': ']'}
        symbols: List[str] = []
        while not self.eof:
            if len(symbols) == 0 and self.current_char in end:
                break
            if self.current_char in brackets:
                symbols.append(brackets[self.current_char])
            elif len(symbols) > 0 and self.current_char == symbols[-1]:
                symbols.pop()
            self.pos += 1
        if len(end) > 0 and self.eof:
            self.fail('Could not find end of expression starting at %d.' % startPos)
        value = self.definition[startPos:self.pos].strip()
    return ASTFallbackExpr(value.strip())