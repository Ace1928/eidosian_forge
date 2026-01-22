import re
import string
import weakref
from string import whitespace
from types import MethodType
from .constants import DefaultValue
from .trait_base import Undefined, Uninitialized
from .trait_errors import TraitError
from .trait_notifiers import TraitChangeNotifyWrapper
from .util.weakiddict import WeakIDKeyDict
class ListenerParser:

    @property
    def next(self):
        """The next character from the string being parsed."""
        index = self.index
        self.index += 1
        if index >= self.len_text:
            return EOS
        return self.text[index]

    @property
    def backspace(self):
        """Backspaces to the last character processed."""
        self.index = max(0, self.index - 1)

    @property
    def skip_ws(self):
        """The next non-whitespace character."""
        while True:
            c = self.next
            if c not in whitespace:
                return c

    @property
    def name(self):
        """The next Python attribute name within the string."""
        match = name_pat.match(self.text, self.index - 1)
        if match is None:
            return ''
        self.index = match.start(2)
        return match.group(1)

    def __init__(self, text, *, handler=None, wrapped_handler_ref=None, dispatch='', priority=False, deferred=False, handler_type=ANY_LISTENER):
        self.text = text
        self.len_text = len(self.text)
        self.index = 0
        self.handler = handler
        self.wrapped_handler_ref = wrapped_handler_ref
        self.dispatch = dispatch
        self.priority = priority
        self.listener = self.parse(deferred, handler_type)

    def parse(self, deferred, handler_type):
        """ Parses the text and returns the appropriate collection of
            ListenerBase objects described by the text.

        Parameters
        ----------
        deferred : bool
            Should registering listeners for items reachable from this listener
            item be deferred until the associated trait is first read or set?
        handler_type : int
            The type of handler being used; one of {ANY_LISTENER, SRC_LISTENER,
            DST_LISTENER}.
        """
        if self.text.strip().endswith(','):
            self.error("Error parsing name. Trailing ',' is not allowed")
        match = simple_pat.match(self.text)
        if match is not None:
            return ListenerItem(name=match.group(1), notify=match.group(2) == '.', next=ListenerItem(name=match.group(3), handler=self.handler, wrapped_handler_ref=self.wrapped_handler_ref, dispatch=self.dispatch, priority=self.priority, deferred=False, type=ANY_LISTENER), handler=self.handler, wrapped_handler_ref=self.wrapped_handler_ref, dispatch=self.dispatch, priority=self.priority, deferred=deferred, type=handler_type)
        return self.parse_group(terminator=EOS, deferred=deferred, handler_type=handler_type)

    def parse_group(self, *, terminator, deferred, handler_type):
        """ Parses the contents of a group.

        Parameters
        ----------
        terminator : str or EOS
            Character on which to halt parsing of this item.
        deferred : bool
            Should registering listeners for items reachable from this listener
            item be deferred until the associated trait is first read or set?
        handler_type : int
            The type of handler being used; one of {ANY_LISTENER, SRC_LISTENER,
            DST_LISTENER}.
        """
        items = []
        while True:
            items.append(self.parse_item(terminator=terminator, deferred=deferred, handler_type=handler_type))
            c = self.skip_ws
            if c == terminator:
                break
            if c != ',':
                if terminator == EOS:
                    self.error("Expected ',' or end of string")
                else:
                    self.error("Expected ',' or '%s'" % terminator)
        if len(items) == 1:
            return items[0]
        return ListenerGroup(items=items)

    def parse_item(self, *, terminator, deferred, handler_type):
        """ Parses a single, complete listener item or group string.

        Parameters
        ----------
        terminator : str or EOS
            Character on which to halt parsing of this item.
        deferred : bool
            Should registering listeners for items reachable from this listener
            item be deferred until the associated trait is first read or set?
        handler_type : int
            The type of handler being used; one of {ANY_LISTENER, SRC_LISTENER,
            DST_LISTENER}.
        """
        c = self.skip_ws
        if c == '[':
            result = self.parse_group(terminator=']', deferred=deferred, handler_type=handler_type)
            c = self.skip_ws
        else:
            name = self.name
            if name != '':
                c = self.next
            result = ListenerItem(name=name, handler=self.handler, wrapped_handler_ref=self.wrapped_handler_ref, dispatch=self.dispatch, priority=self.priority, deferred=deferred, type=handler_type)
            if c in '+-':
                result.name += '*'
                result.metadata_defined = c == '+'
                cn = self.skip_ws
                result.metadata_name = metadata = self.name
                if metadata != '':
                    cn = self.skip_ws
                result.is_anytrait = c == '-' and name == '' and (metadata == '')
                c = cn
                if result.is_anytrait and (not (c == terminator or (c == ',' and terminator == ']'))):
                    self.error('Expected end of name')
            elif c == '?':
                if len(name) == 0:
                    self.error("Expected non-empty name preceding '?'")
                result.name += '?'
                c = self.skip_ws
        cycle = c == '*'
        if cycle:
            c = self.skip_ws
        if c in '.:':
            result.set_notify(c == '.')
            next = self.parse_item(terminator=terminator, deferred=False, handler_type=ANY_LISTENER)
            if cycle:
                last = result
                while last.next is not None:
                    last = last.next
                lg = ListenerGroup(items=[next, result])
                last.set_next(lg)
                result = lg
            else:
                result.set_next(next)
            return result
        if c == '[':
            is_closing_bracket = self.skip_ws == ']'
            next_char = self.skip_ws
            item_complete = next_char == terminator or next_char == ','
            if is_closing_bracket and item_complete:
                self.backspace
                result.is_list_handler = True
            else:
                self.error("Expected '[]' at the end of an item")
        else:
            self.backspace
        if cycle:
            result.set_next(result)
        return result

    def parse_metadata(self, item):
        """ Parses the metadata portion of a listener item.
        """
        self.skip_ws
        item.metadata_name = name = self.name
        if name == '':
            self.backspace

    def error(self, msg):
        """ Raises a syntax error.
        """
        raise TraitError("%s at column %d of '%s'" % (msg, self.index, self.text))