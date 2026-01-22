from copy import copy, deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterator, Optional, Set, Tuple, Union
import interegular
from interegular.fsm import FSM
from interegular.patterns import Unsupported
from lark import Lark, Token
from lark.common import LexerConf, ParserConf
from lark.exceptions import LexError, UnexpectedInput
from lark.indenter import Indenter
from lark.lexer import (
from lark.parser_frontends import (
from lark.parsers.lalr_analysis import (
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser import LALR_Parser, ParseConf, ParserState, _Parser
from outlines.fsm.regex import (
class PartialParserState(ParserState):
    __slots__ = 'use_value_stack'

    def __init__(self, parse_conf, lexer, state_stack=None, value_stack=None, use_value_stack=False):
        super().__init__(parse_conf, lexer, state_stack=state_stack, value_stack=value_stack)
        self.use_value_stack = use_value_stack

    def feed_token(self, token, is_end=False):
        if token.type == 'partial':
            current_state = self.state_stack[-1]
            current_lexer = get_contextual_lexer(self.lexer).lexers[current_state]
            can_transition = False
            for terminal_info in token.value.terminals_and_info:
                if terminal_info.terminal_name not in current_lexer.ignore_types:
                    test_token = Token.new_borrow_pos(terminal_info.terminal_name, '', token)
                    stack = copy(self.state_stack)
                    try:
                        self.feed_token_no_stack(test_token, is_end=is_end)
                        can_transition = True
                        break
                    except UnexpectedToken:
                        continue
                    finally:
                        self.state_stack = stack
                else:
                    can_transition = True
            if not can_transition:
                expected = {s for s in self.parse_conf.states[current_state].keys() if s.isupper()}
                raise UnexpectedToken(token, expected, state=self, interactive_parser=None)
        elif self.use_value_stack:
            super().feed_token(token, is_end=is_end)
        else:
            self.feed_token_no_stack(token, is_end=is_end)

    def feed_token_no_stack(self, token, is_end=False):
        """
        This is a copy of `ParserState.feed_token` with all the value stack
        steps removed.  Since we're not exactly parsing in order to obtain a
        CST or anything similar, we can avoid the growing expense of tracking
        the parse tree.
        """
        state_stack = self.state_stack
        states = self.parse_conf.states
        end_state = self.parse_conf.end_state
        while True:
            state = state_stack[-1]
            try:
                action, arg = states[state][token.type]
            except KeyError:
                expected = {s for s in states[state].keys() if s.isupper()}
                raise UnexpectedToken(token, expected, state=self, interactive_parser=None)
            assert arg != end_state
            if action is Shift:
                assert not is_end
                state_stack.append(arg)
                return
            else:
                rule = arg
                size = len(rule.expansion)
                if size:
                    del state_stack[-size:]
                _action, new_state = states[state_stack[-1]][rule.origin.name]
                assert _action is Shift
                state_stack.append(new_state)
                if is_end and state_stack[-1] == end_state:
                    return

    def __copy__(self):
        return type(self)(self.parse_conf, copy(self.lexer), copy(self.state_stack), deepcopy(self.value_stack), use_value_stack=self.use_value_stack)

    def __repr__(self):
        return f'{type(self).__name__}(lexer={self.lexer!r}, state_stack={self.state_stack!r})'