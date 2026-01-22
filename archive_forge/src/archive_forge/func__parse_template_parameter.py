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
def _parse_template_parameter(self) -> ASTTemplateParam:
    self.skip_ws()
    if self.skip_word('template'):
        nestedParams = self._parse_template_parameter_list()
    else:
        nestedParams = None
    pos = self.pos
    try:
        key = None
        self.skip_ws()
        if self.skip_word_and_ws('typename'):
            key = 'typename'
        elif self.skip_word_and_ws('class'):
            key = 'class'
        elif nestedParams:
            self.fail("Expected 'typename' or 'class' after template template parameter list.")
        else:
            self.fail("Expected 'typename' or 'class' in the beginning of template type parameter.")
        self.skip_ws()
        parameterPack = self.skip_string('...')
        self.skip_ws()
        if self.match(identifier_re):
            identifier = ASTIdentifier(self.matched_text)
        else:
            identifier = None
        self.skip_ws()
        if not parameterPack and self.skip_string('='):
            default = self._parse_type(named=False, outer=None)
        else:
            default = None
            if self.current_char not in ',>':
                self.fail('Expected "," or ">" after (template) type parameter.')
        data = ASTTemplateKeyParamPackIdDefault(key, identifier, parameterPack, default)
        if nestedParams:
            return ASTTemplateParamTemplateType(nestedParams, data)
        else:
            return ASTTemplateParamType(data)
    except DefinitionError as eType:
        if nestedParams:
            raise
        try:
            self.pos = pos
            param = self._parse_type_with_init('maybe', 'templateParam')
            self.skip_ws()
            parameterPack = self.skip_string('...')
            return ASTTemplateParamNonType(param, parameterPack)
        except DefinitionError as eNonType:
            self.pos = pos
            header = 'Error when parsing template parameter.'
            errs = []
            errs.append((eType, 'If unconstrained type parameter or template type parameter'))
            errs.append((eNonType, 'If constrained type parameter or non-type parameter'))
            raise self._make_multi_error(errs, header)