import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope
def _visit_part(self, part, spacing, leaf):
    value = part.value
    type_ = part.type
    if type_ == 'error_leaf':
        return
    if value == ',' and part.parent.type == 'dictorsetmaker':
        self._indentation_tos = self._indentation_tos.parent
    node = self._indentation_tos
    if type_ == 'comment':
        if value.startswith('##'):
            if value.lstrip('#'):
                self.add_issue(part, 266, "Too many leading '#' for block comment.")
        elif self._on_newline:
            if not re.match('#:? ', value) and (not value == '#') and (not (value.startswith('#!') and part.start_pos == (1, 0))):
                self.add_issue(part, 265, "Block comment should start with '# '")
        elif not re.match('#:? [^ ]', value):
            self.add_issue(part, 262, "Inline comment should start with '# '")
        self._reset_newlines(spacing, leaf, is_comment=True)
    elif type_ == 'newline':
        if self._newline_count > self._get_wanted_blank_lines_count():
            self.add_issue(part, 303, 'Too many blank lines (%s)' % self._newline_count)
        elif leaf in ('def', 'class') and leaf.parent.parent.type == 'decorated':
            self.add_issue(part, 304, 'Blank lines found after function decorator')
        self._newline_count += 1
    if type_ == 'backslash':
        if node.type != IndentationTypes.BACKSLASH:
            if node.type != IndentationTypes.SUITE:
                self.add_issue(part, 502, 'The backslash is redundant between brackets')
            else:
                indentation = node.indentation
                if self._in_suite_introducer and node.type == IndentationTypes.SUITE:
                    indentation += self._config.indentation
                self._indentation_tos = BackslashNode(self._config, indentation, part, spacing, parent=self._indentation_tos)
    elif self._on_newline:
        indentation = spacing.value
        if node.type == IndentationTypes.BACKSLASH and self._previous_part.type == 'newline':
            self._indentation_tos = self._indentation_tos.parent
        if not self._check_tabs_spaces(spacing):
            should_be_indentation = node.indentation
            if type_ == 'comment':
                n = self._last_indentation_tos
                while True:
                    if len(indentation) > len(n.indentation):
                        break
                    should_be_indentation = n.indentation
                    self._last_indentation_tos = n
                    if n == node:
                        break
                    n = n.parent
            if self._new_statement:
                if type_ == 'newline':
                    if indentation:
                        self.add_issue(spacing, 291, 'Trailing whitespace')
                elif indentation != should_be_indentation:
                    s = '%s %s' % (len(self._config.indentation), self._indentation_type)
                    self.add_issue(part, 111, 'Indentation is not a multiple of ' + s)
            else:
                if value in '])}':
                    should_be_indentation = node.bracket_indentation
                else:
                    should_be_indentation = node.indentation
                if self._in_suite_introducer and indentation == node.get_latest_suite_node().indentation + self._config.indentation:
                    self.add_issue(part, 129, 'Line with same indent as next logical block')
                elif indentation != should_be_indentation:
                    if not self._check_tabs_spaces(spacing) and part.value not in {'\n', '\r\n', '\r'}:
                        if value in '])}':
                            if node.type == IndentationTypes.VERTICAL_BRACKET:
                                self.add_issue(part, 124, 'Closing bracket does not match visual indentation')
                            else:
                                self.add_issue(part, 123, "Losing bracket does not match indentation of opening bracket's line")
                        elif len(indentation) < len(should_be_indentation):
                            if node.type == IndentationTypes.VERTICAL_BRACKET:
                                self.add_issue(part, 128, 'Continuation line under-indented for visual indent')
                            elif node.type == IndentationTypes.BACKSLASH:
                                self.add_issue(part, 122, 'Continuation line missing indentation or outdented')
                            elif node.type == IndentationTypes.IMPLICIT:
                                self.add_issue(part, 135, 'xxx')
                            else:
                                self.add_issue(part, 121, 'Continuation line under-indented for hanging indent')
                        elif node.type == IndentationTypes.VERTICAL_BRACKET:
                            self.add_issue(part, 127, 'Continuation line over-indented for visual indent')
                        elif node.type == IndentationTypes.IMPLICIT:
                            self.add_issue(part, 136, 'xxx')
                        else:
                            self.add_issue(part, 126, 'Continuation line over-indented for hanging indent')
    else:
        self._check_spacing(part, spacing)
    self._check_line_length(part, spacing)
    if value and value in '()[]{}' and (type_ != 'error_leaf') and (part.parent.type != 'error_node'):
        if value in _OPENING_BRACKETS:
            self._indentation_tos = BracketNode(self._config, part, parent=self._indentation_tos, in_suite_introducer=self._in_suite_introducer)
        else:
            assert node.type != IndentationTypes.IMPLICIT
            self._indentation_tos = self._indentation_tos.parent
    elif value in ('=', ':') and self._implicit_indentation_possible and (part.parent.type in _IMPLICIT_INDENTATION_TYPES):
        indentation = node.indentation
        self._indentation_tos = ImplicitNode(self._config, part, parent=self._indentation_tos)
    self._on_newline = type_ in ('newline', 'backslash', 'bom')
    self._previous_part = part
    self._previous_spacing = spacing