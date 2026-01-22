from typing import Dict, Any, Optional
from ..lexer import Token, LexerThread
from ..utils import Serialize
from ..common import ParserConf, ParserCallbacks
from .lalr_analysis import LALR_Analyzer, IntParseTable, ParseTableBase
from .lalr_interactive_parser import InteractiveParser
from lark.exceptions import UnexpectedCharacters, UnexpectedInput, UnexpectedToken
from .lalr_parser_state import ParserState, ParseConf
class LALR_Parser(Serialize):

    def __init__(self, parser_conf: ParserConf, debug: bool=False, strict: bool=False):
        analysis = LALR_Analyzer(parser_conf, debug=debug, strict=strict)
        analysis.compute_lalr()
        callbacks = parser_conf.callbacks
        self._parse_table = analysis.parse_table
        self.parser_conf = parser_conf
        self.parser = _Parser(analysis.parse_table, callbacks, debug)

    @classmethod
    def deserialize(cls, data, memo, callbacks, debug=False):
        inst = cls.__new__(cls)
        inst._parse_table = IntParseTable.deserialize(data, memo)
        inst.parser = _Parser(inst._parse_table, callbacks, debug)
        return inst

    def serialize(self, memo: Any=None) -> Dict[str, Any]:
        return self._parse_table.serialize(memo)

    def parse_interactive(self, lexer: LexerThread, start: str):
        return self.parser.parse(lexer, start, start_interactive=True)

    def parse(self, lexer, start, on_error=None):
        try:
            return self.parser.parse(lexer, start)
        except UnexpectedInput as e:
            if on_error is None:
                raise
            while True:
                if isinstance(e, UnexpectedCharacters):
                    s = e.interactive_parser.lexer_thread.state
                    p = s.line_ctr.char_pos
                if not on_error(e):
                    raise e
                if isinstance(e, UnexpectedCharacters):
                    if p == s.line_ctr.char_pos:
                        s.line_ctr.feed(s.text[p:p + 1])
                try:
                    return e.interactive_parser.resume_parse()
                except UnexpectedToken as e2:
                    if isinstance(e, UnexpectedToken) and e.token.type == e2.token.type == '$END' and (e.interactive_parser == e2.interactive_parser):
                        raise e2
                    e = e2
                except UnexpectedCharacters as e2:
                    e = e2