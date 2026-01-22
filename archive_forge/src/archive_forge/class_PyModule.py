import builtins
import inspect
import re
import sys
import typing
import warnings
from inspect import Parameter
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Type, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import Inliner
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref, pending_xref_condition
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, Index, IndexEntry, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.pycode.ast import ast
from sphinx.pycode.ast import parse as ast_parse
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective, switch_source_input
from sphinx.util.inspect import signature_from_str
from sphinx.util.nodes import (find_pending_xref_condition, make_id, make_refnode,
from sphinx.util.typing import OptionSpec, TextlikeNode
class PyModule(SphinxDirective):
    """
    Directive to mark description of a new module.
    """
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec: OptionSpec = {'platform': lambda x: x, 'synopsis': lambda x: x, 'noindex': directives.flag, 'nocontentsentry': directives.flag, 'deprecated': directives.flag}

    def run(self) -> List[Node]:
        domain = cast(PythonDomain, self.env.get_domain('py'))
        modname = self.arguments[0].strip()
        noindex = 'noindex' in self.options
        self.env.ref_context['py:module'] = modname
        content_node: Element = nodes.section()
        with switch_source_input(self.state, self.content):
            content_node.document = self.state.document
            nested_parse_with_titles(self.state, self.content, content_node)
        ret: List[Node] = []
        if not noindex:
            node_id = make_id(self.env, self.state.document, 'module', modname)
            target = nodes.target('', '', ids=[node_id], ismod=True)
            self.set_source_info(target)
            self.state.document.note_explicit_target(target)
            domain.note_module(modname, node_id, self.options.get('synopsis', ''), self.options.get('platform', ''), 'deprecated' in self.options)
            domain.note_object(modname, 'module', node_id, location=target)
            ret.append(target)
            indextext = '%s; %s' % (pairindextypes['module'], modname)
            inode = addnodes.index(entries=[('pair', indextext, node_id, '', None)])
            ret.append(inode)
        ret.extend(content_node.children)
        return ret

    def make_old_id(self, name: str) -> str:
        """Generate old styled node_id.

        Old styled node_id is incompatible with docutils' node_id.
        It can contain dots and hyphens.

        .. note:: Old styled node_id was mainly used until Sphinx-3.0.
        """
        return 'module-%s' % name