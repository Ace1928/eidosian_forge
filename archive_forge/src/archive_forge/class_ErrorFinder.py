import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
class ErrorFinder(Normalizer):
    """
    Searches for errors in the syntax tree.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._error_dict = {}
        self.version = self.grammar.version_info

    def initialize(self, node):

        def create_context(node):
            if node is None:
                return None
            parent_context = create_context(node.parent)
            if node.type in ('classdef', 'funcdef', 'file_input'):
                return _Context(node, self._add_syntax_error, parent_context)
            return parent_context
        self.context = create_context(node) or _Context(node, self._add_syntax_error)
        self._indentation_count = 0

    def visit(self, node):
        if node.type == 'error_node':
            with self.visit_node(node):
                return ''
        return super().visit(node)

    @contextmanager
    def visit_node(self, node):
        self._check_type_rules(node)
        if node.type in _BLOCK_STMTS:
            with self.context.add_block(node):
                if len(self.context.blocks) == _MAX_BLOCK_SIZE:
                    self._add_syntax_error(node, 'too many statically nested blocks')
                yield
            return
        elif node.type == 'suite':
            self._indentation_count += 1
            if self._indentation_count == _MAX_INDENT_COUNT:
                self._add_indentation_error(node.children[1], 'too many levels of indentation')
        yield
        if node.type == 'suite':
            self._indentation_count -= 1
        elif node.type in ('classdef', 'funcdef'):
            context = self.context
            self.context = context.parent_context
            self.context.close_child_context(context)

    def visit_leaf(self, leaf):
        if leaf.type == 'error_leaf':
            if leaf.token_type in ('INDENT', 'ERROR_DEDENT'):
                spacing = list(leaf.get_next_leaf()._split_prefix())[-1]
                if leaf.token_type == 'INDENT':
                    message = 'unexpected indent'
                else:
                    message = 'unindent does not match any outer indentation level'
                self._add_indentation_error(spacing, message)
            else:
                if leaf.value.startswith('\\'):
                    message = 'unexpected character after line continuation character'
                else:
                    match = re.match('\\w{,2}("{1,3}|\'{1,3})', leaf.value)
                    if match is None:
                        message = 'invalid syntax'
                        if self.version >= (3, 9) and leaf.value in _get_token_collection(self.version).always_break_tokens:
                            message = 'f-string: ' + message
                    elif len(match.group(1)) == 1:
                        message = 'EOL while scanning string literal'
                    else:
                        message = 'EOF while scanning triple-quoted string literal'
                self._add_syntax_error(leaf, message)
            return ''
        elif leaf.value == ':':
            parent = leaf.parent
            if parent.type in ('classdef', 'funcdef'):
                self.context = self.context.add_context(parent)
        return super().visit_leaf(leaf)

    def _add_indentation_error(self, spacing, message):
        self.add_issue(spacing, 903, 'IndentationError: ' + message)

    def _add_syntax_error(self, node, message):
        self.add_issue(node, 901, 'SyntaxError: ' + message)

    def add_issue(self, node, code, message):
        line = node.start_pos[0]
        args = (code, message, node)
        self._error_dict.setdefault(line, args)

    def finalize(self):
        self.context.finalize()
        for code, message, node in self._error_dict.values():
            self.issues.append(Issue(node, code, message))