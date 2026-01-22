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
class CXRefRole(XRefRole):

    def process_link(self, env: BuildEnvironment, refnode: Element, has_explicit_title: bool, title: str, target: str) -> Tuple[str, str]:
        refnode.attributes.update(env.ref_context)
        if not has_explicit_title:
            title = anon_identifier_re.sub('[anonymous]', str(title))
        if not has_explicit_title:
            target = target.lstrip('~')
            if title[0:1] == '~':
                title = title[1:]
                dot = title.rfind('.')
                if dot != -1:
                    title = title[dot + 1:]
        return (title, target)

    def run(self) -> Tuple[List[Node], List[system_message]]:
        if not self.env.config['c_allow_pre_v3'] or self.disabled:
            return super().run()
        text = self.text.replace('\n', ' ')
        parser = DefinitionParser(text, location=self.get_location(), config=self.env.config)
        try:
            parser.parse_xref_object()
            return super().run()
        except DefinitionError as eOrig:
            parser.pos = 0
            try:
                ast = parser.parse_expression()
            except DefinitionError:
                return super().run()
            classes = ['xref', 'c', 'c-texpr']
            parentSymbol = self.env.temp_data.get('cpp:parent_symbol', None)
            if parentSymbol is None:
                parentSymbol = self.env.domaindata['c']['root_symbol']
            signode = nodes.inline(classes=classes)
            ast.describe_signature(signode, 'markType', self.env, parentSymbol)
            if self.env.config['c_warn_on_allowed_pre_v3']:
                msg = "{}: Pre-v3 C type role ':c:type:`{}`' converted to ':c:expr:`{}`'."
                msg += '\nThe original parsing error was:\n{}'
                msg = msg.format(RemovedInSphinx60Warning.__name__, text, text, eOrig)
                logger.warning(msg, location=self.get_location())
            return ([signode], [])