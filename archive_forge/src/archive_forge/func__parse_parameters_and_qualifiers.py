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
def _parse_parameters_and_qualifiers(self, paramMode: str) -> ASTParametersQualifiers:
    if paramMode == 'new':
        return None
    self.skip_ws()
    if not self.skip_string('('):
        if paramMode == 'function':
            self.fail('Expecting "(" in parameters-and-qualifiers.')
        else:
            return None
    args = []
    self.skip_ws()
    if not self.skip_string(')'):
        while 1:
            self.skip_ws()
            if self.skip_string('...'):
                args.append(ASTFunctionParameter(None, True))
                self.skip_ws()
                if not self.skip_string(')'):
                    self.fail('Expected ")" after "..." in parameters-and-qualifiers.')
                break
            arg = self._parse_type_with_init(outer=None, named='single')
            args.append(ASTFunctionParameter(arg))
            self.skip_ws()
            if self.skip_string(','):
                continue
            elif self.skip_string(')'):
                break
            else:
                self.fail('Expecting "," or ")" in parameters-and-qualifiers, got "%s".' % self.current_char)
    self.skip_ws()
    const = self.skip_word_and_ws('const')
    volatile = self.skip_word_and_ws('volatile')
    if not const:
        const = self.skip_word_and_ws('const')
    refQual = None
    if self.skip_string('&&'):
        refQual = '&&'
    if not refQual and self.skip_string('&'):
        refQual = '&'
    exceptionSpec = None
    self.skip_ws()
    if self.skip_string('noexcept'):
        if self.skip_string_and_ws('('):
            expr = self._parse_constant_expression(False)
            self.skip_ws()
            if not self.skip_string(')'):
                self.fail("Expecting ')' to end 'noexcept'.")
            exceptionSpec = ASTNoexceptSpec(expr)
        else:
            exceptionSpec = ASTNoexceptSpec(None)
    self.skip_ws()
    if self.skip_string('->'):
        trailingReturn = self._parse_type(named=False)
    else:
        trailingReturn = None
    self.skip_ws()
    override = self.skip_word_and_ws('override')
    final = self.skip_word_and_ws('final')
    if not override:
        override = self.skip_word_and_ws('override')
    attrs = self._parse_attribute_list()
    self.skip_ws()
    initializer = None
    if paramMode == 'function' and self.skip_string('='):
        self.skip_ws()
        valid = ('0', 'delete', 'default')
        for w in valid:
            if self.skip_word_and_ws(w):
                initializer = w
                break
        if not initializer:
            self.fail('Expected "%s" in initializer-specifier.' % '" or "'.join(valid))
    return ASTParametersQualifiers(args, volatile, const, refQual, exceptionSpec, trailingReturn, override, final, attrs, initializer)