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
class RSTState(StateWS):
    """
    reStructuredText State superclass.

    Contains methods used by all State subclasses.
    """
    nested_sm = NestedStateMachine
    nested_sm_cache = []

    def __init__(self, state_machine, debug=False):
        self.nested_sm_kwargs = {'state_classes': state_classes, 'initial_state': 'Body'}
        StateWS.__init__(self, state_machine, debug)

    def runtime_init(self):
        StateWS.runtime_init(self)
        memo = self.state_machine.memo
        self.memo = memo
        self.reporter = memo.reporter
        self.inliner = memo.inliner
        self.document = memo.document
        self.parent = self.state_machine.node
        if not hasattr(self.reporter, 'get_source_and_line'):
            self.reporter.get_source_and_line = self.state_machine.get_source_and_line

    def goto_line(self, abs_line_offset):
        """
        Jump to input line `abs_line_offset`, ignoring jumps past the end.
        """
        try:
            self.state_machine.goto_line(abs_line_offset)
        except EOFError:
            pass

    def no_match(self, context, transitions):
        """
        Override `StateWS.no_match` to generate a system message.

        This code should never be run.
        """
        self.reporter.severe('Internal error: no transition pattern match.  State: "%s"; transitions: %s; context: %s; current line: %r.' % (self.__class__.__name__, transitions, context, self.state_machine.line))
        return (context, None, [])

    def bof(self, context):
        """Called at beginning of file."""
        return ([], [])

    def nested_parse(self, block, input_offset, node, match_titles=False, state_machine_class=None, state_machine_kwargs=None):
        """
        Create a new StateMachine rooted at `node` and run it over the input
        `block`.
        """
        use_default = 0
        if state_machine_class is None:
            state_machine_class = self.nested_sm
            use_default += 1
        if state_machine_kwargs is None:
            state_machine_kwargs = self.nested_sm_kwargs
            use_default += 1
        block_length = len(block)
        state_machine = None
        if use_default == 2:
            try:
                state_machine = self.nested_sm_cache.pop()
            except IndexError:
                pass
        if not state_machine:
            state_machine = state_machine_class(debug=self.debug, **state_machine_kwargs)
        state_machine.run(block, input_offset, memo=self.memo, node=node, match_titles=match_titles)
        if use_default == 2:
            self.nested_sm_cache.append(state_machine)
        else:
            state_machine.unlink()
        new_offset = state_machine.abs_line_offset()
        if block.parent and len(block) - block_length != 0:
            self.state_machine.next_line(len(block) - block_length)
        return new_offset

    def nested_list_parse(self, block, input_offset, node, initial_state, blank_finish, blank_finish_state=None, extra_settings={}, match_titles=False, state_machine_class=None, state_machine_kwargs=None):
        """
        Create a new StateMachine rooted at `node` and run it over the input
        `block`. Also keep track of optional intermediate blank lines and the
        required final one.
        """
        if state_machine_class is None:
            state_machine_class = self.nested_sm
        if state_machine_kwargs is None:
            state_machine_kwargs = self.nested_sm_kwargs.copy()
        state_machine_kwargs['initial_state'] = initial_state
        state_machine = state_machine_class(debug=self.debug, **state_machine_kwargs)
        if blank_finish_state is None:
            blank_finish_state = initial_state
        state_machine.states[blank_finish_state].blank_finish = blank_finish
        for key, value in list(extra_settings.items()):
            setattr(state_machine.states[initial_state], key, value)
        state_machine.run(block, input_offset, memo=self.memo, node=node, match_titles=match_titles)
        blank_finish = state_machine.states[blank_finish_state].blank_finish
        state_machine.unlink()
        return (state_machine.abs_line_offset(), blank_finish)

    def section(self, title, source, style, lineno, messages):
        """Check for a valid subsection and create one if it checks out."""
        if self.check_subsection(source, style, lineno):
            self.new_subsection(title, lineno, messages)

    def check_subsection(self, source, style, lineno):
        """
        Check for a valid subsection header.  Return 1 (true) or None (false).

        When a new section is reached that isn't a subsection of the current
        section, back up the line count (use ``previous_line(-x)``), then
        ``raise EOFError``.  The current StateMachine will finish, then the
        calling StateMachine can re-examine the title.  This will work its way
        back up the calling chain until the correct section level isreached.

        @@@ Alternative: Evaluate the title, store the title info & level, and
        back up the chain until that level is reached.  Store in memo? Or
        return in results?

        :Exception: `EOFError` when a sibling or supersection encountered.
        """
        memo = self.memo
        title_styles = memo.title_styles
        mylevel = memo.section_level
        try:
            level = title_styles.index(style) + 1
        except ValueError:
            if len(title_styles) == memo.section_level:
                title_styles.append(style)
                return 1
            else:
                self.parent += self.title_inconsistent(source, lineno)
                return None
        if level <= mylevel:
            memo.section_level = level
            if len(style) == 2:
                memo.section_bubble_up_kludge = True
            self.state_machine.previous_line(len(style) + 1)
            raise EOFError
        if level == mylevel + 1:
            return 1
        else:
            self.parent += self.title_inconsistent(source, lineno)
            return None

    def title_inconsistent(self, sourcetext, lineno):
        error = self.reporter.severe('Title level inconsistent:', nodes.literal_block('', sourcetext), line=lineno)
        return error

    def new_subsection(self, title, lineno, messages):
        """Append new subsection to document tree. On return, check level."""
        memo = self.memo
        mylevel = memo.section_level
        memo.section_level += 1
        section_node = nodes.section()
        self.parent += section_node
        textnodes, title_messages = self.inline_text(title, lineno)
        titlenode = nodes.title(title, '', *textnodes)
        name = normalize_name(titlenode.astext())
        section_node['names'].append(name)
        section_node += titlenode
        section_node += messages
        section_node += title_messages
        self.document.note_implicit_target(section_node, section_node)
        offset = self.state_machine.line_offset + 1
        absoffset = self.state_machine.abs_line_offset() + 1
        newabsoffset = self.nested_parse(self.state_machine.input_lines[offset:], input_offset=absoffset, node=section_node, match_titles=True)
        self.goto_line(newabsoffset)
        if memo.section_level <= mylevel:
            raise EOFError
        memo.section_level = mylevel

    def paragraph(self, lines, lineno):
        """
        Return a list (paragraph & messages) & a boolean: literal_block next?
        """
        data = '\n'.join(lines).rstrip()
        if re.search('(?<!\\\\)(\\\\\\\\)*::$', data):
            if len(data) == 2:
                return ([], 1)
            elif data[-3] in ' \n':
                text = data[:-3].rstrip()
            else:
                text = data[:-1]
            literalnext = 1
        else:
            text = data
            literalnext = 0
        textnodes, messages = self.inline_text(text, lineno)
        p = nodes.paragraph(data, '', *textnodes)
        p.source, p.line = self.state_machine.get_source_and_line(lineno)
        return ([p] + messages, literalnext)

    def inline_text(self, text, lineno):
        """
        Return 2 lists: nodes (text and inline elements), and system_messages.
        """
        nodes, messages = self.inliner.parse(text, lineno, self.memo, self.parent)
        return (nodes, messages)

    def unindent_warning(self, node_name):
        lineno = self.state_machine.abs_line_number() + 1
        return self.reporter.warning('%s ends without a blank line; unexpected unindent.' % node_name, line=lineno)