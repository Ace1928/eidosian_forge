from typing import Dict, Any, Optional
from ..lexer import Token, LexerThread
from ..utils import Serialize
from ..common import ParserConf, ParserCallbacks
from .lalr_analysis import LALR_Analyzer, IntParseTable, ParseTableBase
from .lalr_interactive_parser import InteractiveParser
from lark.exceptions import UnexpectedCharacters, UnexpectedInput, UnexpectedToken
from .lalr_parser_state import ParserState, ParseConf
class _Parser:
    parse_table: ParseTableBase
    callbacks: ParserCallbacks
    debug: bool

    def __init__(self, parse_table: ParseTableBase, callbacks: ParserCallbacks, debug: bool=False):
        self.parse_table = parse_table
        self.callbacks = callbacks
        self.debug = debug

    def parse(self, lexer: LexerThread, start: str, value_stack=None, state_stack=None, start_interactive=False):
        parse_conf = ParseConf(self.parse_table, self.callbacks, start)
        parser_state = ParserState(parse_conf, lexer, state_stack, value_stack)
        if start_interactive:
            return InteractiveParser(self, parser_state, parser_state.lexer)
        return self.parse_from_state(parser_state)

    def parse_from_state(self, state: ParserState, last_token: Optional[Token]=None):
        """Run the main LALR parser loop

        Parameters:
            state - the initial state. Changed in-place.
            last_token - Used only for line information in case of an empty lexer.
        """
        try:
            token = last_token
            for token in state.lexer.lex(state):
                assert token is not None
                state.feed_token(token)
            end_token = Token.new_borrow_pos('$END', '', token) if token else Token('$END', '', 0, 1, 1)
            return state.feed_token(end_token, True)
        except UnexpectedInput as e:
            try:
                e.interactive_parser = InteractiveParser(self, state, state.lexer)
            except NameError:
                pass
            raise e
        except Exception as e:
            if self.debug:
                print('')
                print('STATE STACK DUMP')
                print('----------------')
                for i, s in enumerate(state.state_stack):
                    print('%d)' % i, s)
                print('')
            raise