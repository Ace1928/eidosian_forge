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
def _parse_declarator_name_suffix(self, named: Union[bool, str], paramMode: str, typed: bool) -> ASTDeclarator:
    assert named in (True, False, 'single')
    if named == 'single':
        if self.match(identifier_re):
            if self.matched_text in _keywords:
                self.fail('Expected identifier, got keyword: %s' % self.matched_text)
            if self.matched_text in self.config.c_extra_keywords:
                msg = 'Expected identifier, got user-defined keyword: %s.' + ' Remove it from c_extra_keywords to allow it as identifier.\n' + 'Currently c_extra_keywords is %s.'
                self.fail(msg % (self.matched_text, str(self.config.c_extra_keywords)))
            identifier = ASTIdentifier(self.matched_text)
            declId = ASTNestedName([identifier], rooted=False)
        else:
            declId = None
    elif named:
        declId = self._parse_nested_name()
    else:
        declId = None
    arrayOps = []
    while 1:
        self.skip_ws()
        if typed and self.skip_string('['):
            self.skip_ws()
            static = False
            const = False
            volatile = False
            restrict = False
            while True:
                if not static:
                    if self.skip_word_and_ws('static'):
                        static = True
                        continue
                if not const:
                    if self.skip_word_and_ws('const'):
                        const = True
                        continue
                if not volatile:
                    if self.skip_word_and_ws('volatile'):
                        volatile = True
                        continue
                if not restrict:
                    if self.skip_word_and_ws('restrict'):
                        restrict = True
                        continue
                break
            vla = False if static else self.skip_string_and_ws('*')
            if vla:
                if not self.skip_string(']'):
                    self.fail("Expected ']' in end of array operator.")
                size = None
            elif self.skip_string(']'):
                size = None
            else:

                def parser():
                    return self._parse_expression()
                size = self._parse_expression_fallback([']'], parser)
                self.skip_ws()
                if not self.skip_string(']'):
                    self.fail("Expected ']' in end of array operator.")
            arrayOps.append(ASTArray(static, const, volatile, restrict, vla, size))
        else:
            break
    param = self._parse_parameters(paramMode)
    if param is None and len(arrayOps) == 0:
        if named and paramMode == 'type' and typed:
            self.skip_ws()
            if self.skip_string(':'):
                size = self._parse_constant_expression()
                return ASTDeclaratorNameBitField(declId=declId, size=size)
    return ASTDeclaratorNameParam(declId=declId, arrayOps=arrayOps, param=param)