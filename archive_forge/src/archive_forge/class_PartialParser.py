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
class PartialParser(_Parser):

    def __init__(self, parse_table, callbacks, debug=False, use_value_stack=False):
        super().__init__(parse_table, callbacks, debug=debug)
        self.use_value_stack = use_value_stack

    def parse(self, lexer, start, value_stack=None, state_stack=None, start_interactive=False):
        parse_conf = ParseConf(self.parse_table, self.callbacks, start)
        parser_state = PartialParserState(parse_conf, copy(lexer), state_stack, value_stack, self.use_value_stack)
        if start_interactive:
            return InteractiveParser(self, parser_state, parser_state.lexer)
        return self.parse_from_state(parser_state)

    def parse_from_state(self, state, last_token=None, is_end=False):
        try:
            token = last_token
            for token in state.lexer.lex(state):
                state.feed_token(token)
            if is_end and (not token or token.type != 'partial'):
                end_token = Token.new_borrow_pos('$END', '', token) if token else Token('$END', '', 0, 1, 1)
                state.feed_token(end_token, True)
            return state
        except UnexpectedInput as e:
            try:
                e.interactive_parser = InteractiveParser(self, state, state.lexer)
            except NameError:
                pass
            raise e
        except Exception:
            if self.debug:
                print('')
                print('STATE STACK DUMP')
                print('----------------')
                for i, s in enumerate(state.state_stack):
                    print('%d)' % i, s)
                print('')
            raise