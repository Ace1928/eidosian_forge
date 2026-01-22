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
def _parse_decl_specs_simple(self, outer: str, typed: bool) -> ASTDeclSpecsSimple:
    """Just parse the simple ones."""
    storage = None
    threadLocal = None
    inline = None
    restrict = None
    volatile = None
    const = None
    attrs = []
    while 1:
        self.skip_ws()
        if not storage:
            if outer == 'member':
                if self.skip_word('auto'):
                    storage = 'auto'
                    continue
                if self.skip_word('register'):
                    storage = 'register'
                    continue
            if outer in ('member', 'function'):
                if self.skip_word('static'):
                    storage = 'static'
                    continue
                if self.skip_word('extern'):
                    storage = 'extern'
                    continue
        if outer == 'member' and (not threadLocal):
            if self.skip_word('thread_local'):
                threadLocal = 'thread_local'
                continue
            if self.skip_word('_Thread_local'):
                threadLocal = '_Thread_local'
                continue
        if outer == 'function' and (not inline):
            inline = self.skip_word('inline')
            if inline:
                continue
        if not restrict and typed:
            restrict = self.skip_word('restrict')
            if restrict:
                continue
        if not volatile and typed:
            volatile = self.skip_word('volatile')
            if volatile:
                continue
        if not const and typed:
            const = self.skip_word('const')
            if const:
                continue
        attr = self._parse_attribute()
        if attr:
            attrs.append(attr)
            continue
        break
    return ASTDeclSpecsSimple(storage, threadLocal, inline, restrict, volatile, const, ASTAttributeList(attrs))