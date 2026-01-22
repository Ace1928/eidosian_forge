import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
class StateWS(State):
    """
    State superclass specialized for whitespace (blank lines & indents).

    Use this class with `StateMachineWS`.  The transitions 'blank' (for blank
    lines) and 'indent' (for indented text blocks) are added automatically,
    before any other transitions.  The transition method `blank()` handles
    blank lines and `indent()` handles nested indented blocks.  Indented
    blocks trigger a new state machine to be created by `indent()` and run.
    The class of the state machine to be created is in `indent_sm`, and the
    constructor keyword arguments are in the dictionary `indent_sm_kwargs`.

    The methods `known_indent()` and `firstknown_indent()` are provided for
    indented blocks where the indent (all lines' and first line's only,
    respectively) is known to the transition method, along with the attributes
    `known_indent_sm` and `known_indent_sm_kwargs`.  Neither transition method
    is triggered automatically.
    """
    indent_sm = None
    '\n    The `StateMachine` class handling indented text blocks.\n\n    If left as ``None``, `indent_sm` defaults to the value of\n    `State.nested_sm`.  Override it in subclasses to avoid the default.\n    '
    indent_sm_kwargs = None
    '\n    Keyword arguments dictionary, passed to the `indent_sm` constructor.\n\n    If left as ``None``, `indent_sm_kwargs` defaults to the value of\n    `State.nested_sm_kwargs`. Override it in subclasses to avoid the default.\n    '
    known_indent_sm = None
    '\n    The `StateMachine` class handling known-indented text blocks.\n\n    If left as ``None``, `known_indent_sm` defaults to the value of\n    `indent_sm`.  Override it in subclasses to avoid the default.\n    '
    known_indent_sm_kwargs = None
    '\n    Keyword arguments dictionary, passed to the `known_indent_sm` constructor.\n\n    If left as ``None``, `known_indent_sm_kwargs` defaults to the value of\n    `indent_sm_kwargs`. Override it in subclasses to avoid the default.\n    '
    ws_patterns = {'blank': ' *$', 'indent': ' +'}
    'Patterns for default whitespace transitions.  May be overridden in\n    subclasses.'
    ws_initial_transitions = ('blank', 'indent')
    'Default initial whitespace transitions, added before those listed in\n    `State.initial_transitions`.  May be overridden in subclasses.'

    def __init__(self, state_machine, debug=False):
        """
        Initialize a `StateSM` object; extends `State.__init__()`.

        Check for indent state machine attributes, set defaults if not set.
        """
        State.__init__(self, state_machine, debug)
        if self.indent_sm is None:
            self.indent_sm = self.nested_sm
        if self.indent_sm_kwargs is None:
            self.indent_sm_kwargs = self.nested_sm_kwargs
        if self.known_indent_sm is None:
            self.known_indent_sm = self.indent_sm
        if self.known_indent_sm_kwargs is None:
            self.known_indent_sm_kwargs = self.indent_sm_kwargs

    def add_initial_transitions(self):
        """
        Add whitespace-specific transitions before those defined in subclass.

        Extends `State.add_initial_transitions()`.
        """
        State.add_initial_transitions(self)
        if self.patterns is None:
            self.patterns = {}
        self.patterns.update(self.ws_patterns)
        names, transitions = self.make_transitions(self.ws_initial_transitions)
        self.add_transitions(names, transitions)

    def blank(self, match, context, next_state):
        """Handle blank lines. Does nothing. Override in subclasses."""
        return self.nop(match, context, next_state)

    def indent(self, match, context, next_state):
        """
        Handle an indented text block. Extend or override in subclasses.

        Recursively run the registered state machine for indented blocks
        (`self.indent_sm`).
        """
        indented, indent, line_offset, blank_finish = self.state_machine.get_indented()
        sm = self.indent_sm(debug=self.debug, **self.indent_sm_kwargs)
        results = sm.run(indented, input_offset=line_offset)
        return (context, next_state, results)

    def known_indent(self, match, context, next_state):
        """
        Handle a known-indent text block. Extend or override in subclasses.

        Recursively run the registered state machine for known-indent indented
        blocks (`self.known_indent_sm`). The indent is the length of the
        match, ``match.end()``.
        """
        indented, line_offset, blank_finish = self.state_machine.get_known_indented(match.end())
        sm = self.known_indent_sm(debug=self.debug, **self.known_indent_sm_kwargs)
        results = sm.run(indented, input_offset=line_offset)
        return (context, next_state, results)

    def first_known_indent(self, match, context, next_state):
        """
        Handle an indented text block (first line's indent known).

        Extend or override in subclasses.

        Recursively run the registered state machine for known-indent indented
        blocks (`self.known_indent_sm`). The indent is the length of the
        match, ``match.end()``.
        """
        indented, line_offset, blank_finish = self.state_machine.get_first_known_indented(match.end())
        sm = self.known_indent_sm(debug=self.debug, **self.known_indent_sm_kwargs)
        results = sm.run(indented, input_offset=line_offset)
        return (context, next_state, results)