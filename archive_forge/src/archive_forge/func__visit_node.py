import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope
@contextmanager
def _visit_node(self, node):
    typ = node.type
    if typ in 'import_name':
        names = node.get_defined_names()
        if len(names) > 1:
            for name in names[:1]:
                self.add_issue(name, 401, 'Multiple imports on one line')
    elif typ == 'lambdef':
        expr_stmt = node.parent
        if expr_stmt.type == 'expr_stmt' and any((n.type == 'name' for n in expr_stmt.children[:-2:2])):
            self.add_issue(node, 731, 'Do not assign a lambda expression, use a def')
    elif typ == 'try_stmt':
        for child in node.children:
            if child.type == 'keyword' and child.value == 'except':
                self.add_issue(child, 722, 'Do not use bare except, specify exception instead')
    elif typ == 'comparison':
        for child in node.children:
            if child.type not in ('atom_expr', 'power'):
                continue
            if len(child.children) > 2:
                continue
            trailer = child.children[1]
            atom = child.children[0]
            if trailer.type == 'trailer' and atom.type == 'name' and (atom.value == 'type'):
                self.add_issue(node, 721, "Do not compare types, use 'isinstance()")
                break
    elif typ == 'file_input':
        endmarker = node.children[-1]
        prev = endmarker.get_previous_leaf()
        prefix = endmarker.prefix
        if not prefix.endswith('\n') and (not prefix.endswith('\r')) and (prefix or prev is None or prev.value not in {'\n', '\r\n', '\r'}):
            self.add_issue(endmarker, 292, 'No newline at end of file')
    if typ in _IMPORT_TYPES:
        simple_stmt = node.parent
        module = simple_stmt.parent
        if module.type == 'file_input':
            index = module.children.index(simple_stmt)
            for child in module.children[:index]:
                children = [child]
                if child.type == 'simple_stmt':
                    children = child.children[:-1]
                found_docstring = False
                for c in children:
                    if c.type == 'string' and (not found_docstring):
                        continue
                    found_docstring = True
                    if c.type == 'expr_stmt' and all((_is_magic_name(n) for n in c.get_defined_names())):
                        continue
                    if c.type in _IMPORT_TYPES or isinstance(c, Flow):
                        continue
                    self.add_issue(node, 402, 'Module level import not at top of file')
                    break
                else:
                    continue
                break
    implicit_indentation_possible = typ in _IMPLICIT_INDENTATION_TYPES
    in_introducer = typ in _SUITE_INTRODUCERS
    if in_introducer:
        self._in_suite_introducer = True
    elif typ == 'suite':
        if self._indentation_tos.type == IndentationTypes.BACKSLASH:
            self._indentation_tos = self._indentation_tos.parent
        self._indentation_tos = IndentationNode(self._config, self._indentation_tos.indentation + self._config.indentation, parent=self._indentation_tos)
    elif implicit_indentation_possible:
        self._implicit_indentation_possible = True
    yield
    if typ == 'suite':
        assert self._indentation_tos.type == IndentationTypes.SUITE
        self._indentation_tos = self._indentation_tos.parent
        self._wanted_newline_count = None
    elif implicit_indentation_possible:
        self._implicit_indentation_possible = False
        if self._indentation_tos.type == IndentationTypes.IMPLICIT:
            self._indentation_tos = self._indentation_tos.parent
    elif in_introducer:
        self._in_suite_introducer = False
        if typ in ('classdef', 'funcdef'):
            self._wanted_newline_count = self._get_wanted_blank_lines_count()