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
def _parse_literal(self) -> ASTLiteral:
    self.skip_ws()
    if self.skip_word('true'):
        return ASTBooleanLiteral(True)
    if self.skip_word('false'):
        return ASTBooleanLiteral(False)
    pos = self.pos
    if self.match(float_literal_re):
        self.match(float_literal_suffix_re)
        return ASTNumberLiteral(self.definition[pos:self.pos])
    for regex in [binary_literal_re, hex_literal_re, integer_literal_re, octal_literal_re]:
        if self.match(regex):
            self.match(integers_literal_suffix_re)
            return ASTNumberLiteral(self.definition[pos:self.pos])
    string = self._parse_string()
    if string is not None:
        return ASTStringLiteral(string)
    if self.match(char_literal_re):
        prefix = self.last_match.group(1)
        data = self.last_match.group(2)
        try:
            return ASTCharLiteral(prefix, data)
        except UnicodeDecodeError as e:
            self.fail('Can not handle character literal. Internal error was: %s' % e)
        except UnsupportedMultiCharacterCharLiteral:
            self.fail('Can not handle character literal resulting in multiple decoded characters.')
    return None