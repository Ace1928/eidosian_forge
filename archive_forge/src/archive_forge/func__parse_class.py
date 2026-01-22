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
def _parse_class(self) -> ASTClass:
    attrs = self._parse_attribute_list()
    name = self._parse_nested_name()
    self.skip_ws()
    final = self.skip_word_and_ws('final')
    bases = []
    self.skip_ws()
    if self.skip_string(':'):
        while 1:
            self.skip_ws()
            visibility = None
            virtual = False
            pack = False
            if self.skip_word_and_ws('virtual'):
                virtual = True
            if self.match(_visibility_re):
                visibility = self.matched_text
                self.skip_ws()
            if not virtual and self.skip_word_and_ws('virtual'):
                virtual = True
            baseName = self._parse_nested_name()
            self.skip_ws()
            pack = self.skip_string('...')
            bases.append(ASTBaseClass(baseName, visibility, virtual, pack))
            self.skip_ws()
            if self.skip_string(','):
                continue
            else:
                break
    return ASTClass(name, final, bases, attrs)