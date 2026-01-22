import inspect
import os
import posixpath
import re
import sys
import warnings
from inspect import Parameter
from os import path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, cast
from docutils import nodes
from docutils.nodes import Node, system_message
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import RSTStateMachine, Struct, state_classes
from docutils.statemachine import StringList
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.deprecation import (RemovedInSphinx60Warning, RemovedInSphinx70Warning,
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc import INSTANCEATTR, Documenter
from sphinx.ext.autodoc.directive import DocumenterBridge, Options
from sphinx.ext.autodoc.importer import import_module
from sphinx.ext.autodoc.mock import mock
from sphinx.extension import Extension
from sphinx.locale import __
from sphinx.project import Project
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.registry import SphinxComponentRegistry
from sphinx.util import logging, rst
from sphinx.util.docutils import (NullReporter, SphinxDirective, SphinxRole, new_document,
from sphinx.util.inspect import signature_from_str
from sphinx.util.matching import Matcher
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator
class Autosummary(SphinxDirective):
    """
    Pretty table containing short signatures and summaries of functions etc.

    autosummary can also optionally generate a hidden toctree:: node.
    """
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = True
    option_spec: OptionSpec = {'caption': directives.unchanged_required, 'toctree': directives.unchanged, 'nosignatures': directives.flag, 'recursive': directives.flag, 'template': directives.unchanged}

    def run(self) -> List[Node]:
        self.bridge = DocumenterBridge(self.env, self.state.document.reporter, Options(), self.lineno, self.state)
        names = [x.strip().split()[0] for x in self.content if x.strip() and re.search('^[~a-zA-Z_]', x.strip()[0])]
        items = self.get_items(names)
        nodes = self.get_table(items)
        if 'toctree' in self.options:
            dirname = posixpath.dirname(self.env.docname)
            tree_prefix = self.options['toctree'].strip()
            docnames = []
            excluded = Matcher(self.config.exclude_patterns)
            filename_map = self.config.autosummary_filename_map
            for _name, _sig, _summary, real_name in items:
                real_name = filename_map.get(real_name, real_name)
                docname = posixpath.join(tree_prefix, real_name)
                docname = posixpath.normpath(posixpath.join(dirname, docname))
                if docname not in self.env.found_docs:
                    if excluded(self.env.doc2path(docname, False)):
                        msg = __('autosummary references excluded document %r. Ignored.')
                    else:
                        msg = __('autosummary: stub file not found %r. Check your autosummary_generate setting.')
                    logger.warning(msg, real_name, location=self.get_location())
                    continue
                docnames.append(docname)
            if docnames:
                tocnode = addnodes.toctree()
                tocnode['includefiles'] = docnames
                tocnode['entries'] = [(None, docn) for docn in docnames]
                tocnode['maxdepth'] = -1
                tocnode['glob'] = None
                tocnode['caption'] = self.options.get('caption')
                nodes.append(autosummary_toc('', '', tocnode))
        if 'toctree' not in self.options and 'caption' in self.options:
            logger.warning(__('A captioned autosummary requires :toctree: option. ignored.'), location=nodes[-1])
        return nodes

    def import_by_name(self, name: str, prefixes: List[Optional[str]]) -> Tuple[str, Any, Any, str]:
        with mock(self.config.autosummary_mock_imports):
            try:
                return import_by_name(name, prefixes)
            except ImportExceptionGroup as exc:
                try:
                    return import_ivar_by_name(name, prefixes)
                except ImportError as exc2:
                    if exc2.__cause__:
                        errors: List[BaseException] = exc.exceptions + [exc2.__cause__]
                    else:
                        errors = exc.exceptions + [exc2]
                    raise ImportExceptionGroup(exc.args[0], errors)

    def create_documenter(self, app: Sphinx, obj: Any, parent: Any, full_name: str) -> 'Documenter':
        """Get an autodoc.Documenter class suitable for documenting the given
        object.

        Wraps get_documenter and is meant as a hook for extensions.
        """
        doccls = get_documenter(app, obj, parent)
        return doccls(self.bridge, full_name)

    def get_items(self, names: List[str]) -> List[Tuple[str, str, str, str]]:
        """Try to import the given names, and return a list of
        ``[(name, signature, summary_string, real_name), ...]``.
        """
        prefixes = get_import_prefixes_from_env(self.env)
        items: List[Tuple[str, str, str, str]] = []
        max_item_chars = 50
        for name in names:
            display_name = name
            if name.startswith('~'):
                name = name[1:]
                display_name = name.split('.')[-1]
            try:
                real_name, obj, parent, modname = self.import_by_name(name, prefixes=prefixes)
            except ImportExceptionGroup as exc:
                errors = list({'* %s: %s' % (type(e).__name__, e) for e in exc.exceptions})
                logger.warning(__('autosummary: failed to import %s.\nPossible hints:\n%s'), name, '\n'.join(errors), location=self.get_location())
                continue
            self.bridge.result = StringList()
            full_name = real_name
            if not isinstance(obj, ModuleType):
                full_name = modname + '::' + full_name[len(modname) + 1:]
            documenter = self.create_documenter(self.env.app, obj, parent, full_name)
            if not documenter.parse_name():
                logger.warning(__('failed to parse name %s'), real_name, location=self.get_location())
                items.append((display_name, '', '', real_name))
                continue
            if not documenter.import_object():
                logger.warning(__('failed to import object %s'), real_name, location=self.get_location())
                items.append((display_name, '', '', real_name))
                continue
            try:
                documenter.analyzer = ModuleAnalyzer.for_module(documenter.get_real_modname())
                documenter.analyzer.find_attr_docs()
            except PycodeError as err:
                logger.debug('[autodoc] module analyzer failed: %s', err)
                documenter.analyzer = None
            try:
                sig = documenter.format_signature(show_annotation=False)
            except TypeError:
                sig = documenter.format_signature()
            if not sig:
                sig = ''
            else:
                max_chars = max(10, max_item_chars - len(display_name))
                sig = mangle_signature(sig, max_chars=max_chars)
            documenter._extra_indent = ''
            documenter.add_content(None)
            summary = extract_summary(self.bridge.result.data[:], self.state.document)
            items.append((display_name, sig, summary, real_name))
        return items

    def get_table(self, items: List[Tuple[str, str, str, str]]) -> List[Node]:
        """Generate a proper list of table nodes for autosummary:: directive.

        *items* is a list produced by :meth:`get_items`.
        """
        table_spec = addnodes.tabular_col_spec()
        table_spec['spec'] = '\\X{1}{2}\\X{1}{2}'
        table = autosummary_table('')
        real_table = nodes.table('', classes=['autosummary longtable'])
        table.append(real_table)
        group = nodes.tgroup('', cols=2)
        real_table.append(group)
        group.append(nodes.colspec('', colwidth=10))
        group.append(nodes.colspec('', colwidth=90))
        body = nodes.tbody('')
        group.append(body)

        def append_row(*column_texts: str) -> None:
            row = nodes.row('')
            source, line = self.state_machine.get_source_and_line()
            for text in column_texts:
                node = nodes.paragraph('')
                vl = StringList()
                vl.append(text, '%s:%d:<autosummary>' % (source, line))
                with switch_source_input(self.state, vl):
                    self.state.nested_parse(vl, 0, node)
                    try:
                        if isinstance(node[0], nodes.paragraph):
                            node = node[0]
                    except IndexError:
                        pass
                    row.append(nodes.entry('', node))
            body.append(row)
        for name, sig, summary, real_name in items:
            qualifier = 'obj'
            if 'nosignatures' not in self.options:
                col1 = ':py:%s:`%s <%s>`\\ %s' % (qualifier, name, real_name, rst.escape(sig))
            else:
                col1 = ':py:%s:`%s <%s>`' % (qualifier, name, real_name)
            col2 = summary
            append_row(col1, col2)
        return [table_spec, table]