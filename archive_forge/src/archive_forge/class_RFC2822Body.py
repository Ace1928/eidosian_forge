import sys
import re
from types import FunctionType, MethodType
from docutils import nodes, statemachine, utils
from docutils import ApplicationError, DataError
from docutils.statemachine import StateMachineWS, StateWS
from docutils.nodes import fully_normalize_name as normalize_name
from docutils.nodes import whitespace_normalize_name
import docutils.parsers.rst
from docutils.parsers.rst import directives, languages, tableparser, roles
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils import escape2null, unescape, column_width
from docutils.utils import punctuation_chars, roman, urischemes
from docutils.utils import split_escaped_whitespace
class RFC2822Body(Body):
    """
    RFC2822 headers are only valid as the first constructs in documents.  As
    soon as anything else appears, the `Body` state should take over.
    """
    patterns = Body.patterns.copy()
    patterns['rfc2822'] = '[!-9;-~]+:( +|$)'
    initial_transitions = [(name, 'Body') for name in Body.initial_transitions]
    initial_transitions.insert(-1, ('rfc2822', 'Body'))

    def rfc2822(self, match, context, next_state):
        """RFC2822-style field list item."""
        fieldlist = nodes.field_list(classes=['rfc2822'])
        self.parent += fieldlist
        field, blank_finish = self.rfc2822_field(match)
        fieldlist += field
        offset = self.state_machine.line_offset + 1
        newline_offset, blank_finish = self.nested_list_parse(self.state_machine.input_lines[offset:], input_offset=self.state_machine.abs_line_offset() + 1, node=fieldlist, initial_state='RFC2822List', blank_finish=blank_finish)
        self.goto_line(newline_offset)
        if not blank_finish:
            self.parent += self.unindent_warning('RFC2822-style field list')
        return ([], next_state, [])

    def rfc2822_field(self, match):
        name = match.string[:match.string.find(':')]
        indented, indent, line_offset, blank_finish = self.state_machine.get_first_known_indented(match.end(), until_blank=True)
        fieldnode = nodes.field()
        fieldnode += nodes.field_name(name, name)
        fieldbody = nodes.field_body('\n'.join(indented))
        fieldnode += fieldbody
        if indented:
            self.nested_parse(indented, input_offset=line_offset, node=fieldbody)
        return (fieldnode, blank_finish)