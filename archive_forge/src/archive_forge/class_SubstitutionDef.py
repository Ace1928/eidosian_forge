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
class SubstitutionDef(Body):
    """
    Parser for the contents of a substitution_definition element.
    """
    patterns = {'embedded_directive': re.compile('(%s)::( +|$)' % Inliner.simplename, re.UNICODE), 'text': ''}
    initial_transitions = ['embedded_directive', 'text']

    def embedded_directive(self, match, context, next_state):
        nodelist, blank_finish = self.directive(match, alt=self.parent['names'][0])
        self.parent += nodelist
        if not self.state_machine.at_eof():
            self.blank_finish = blank_finish
        raise EOFError

    def text(self, match, context, next_state):
        if not self.state_machine.at_eof():
            self.blank_finish = self.state_machine.is_next_line_blank()
        raise EOFError