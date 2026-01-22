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
class NestedStateMachine(StateMachineWS):
    """
    StateMachine run from within other StateMachine runs, to parse nested
    document structures.
    """

    def run(self, input_lines, input_offset, memo, node, match_titles=True):
        """
        Parse `input_lines` and populate a `docutils.nodes.document` instance.

        Extend `StateMachineWS.run()`: set up document-wide data.
        """
        self.match_titles = match_titles
        self.memo = memo
        self.document = memo.document
        self.attach_observer(self.document.note_source)
        self.reporter = memo.reporter
        self.language = memo.language
        self.node = node
        results = StateMachineWS.run(self, input_lines, input_offset)
        assert results == [], 'NestedStateMachine.run() results should be empty!'
        return results