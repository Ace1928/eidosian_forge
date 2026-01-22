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
class EnumeratedList(SpecializedBody):
    """Second and subsequent enumerated_list list_items."""

    def enumerator(self, match, context, next_state):
        """Enumerated list item."""
        format, sequence, text, ordinal = self.parse_enumerator(match, self.parent['enumtype'])
        if format != self.format or (sequence != '#' and (sequence != self.parent['enumtype'] or self.auto or ordinal != self.lastordinal + 1)) or (not self.is_enumerated_list_item(ordinal, sequence, format)):
            self.invalid_input()
        if sequence == '#':
            self.auto = 1
        listitem, blank_finish = self.list_item(match.end())
        self.parent += listitem
        self.blank_finish = blank_finish
        self.lastordinal = ordinal
        return ([], next_state, [])