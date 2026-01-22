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
def _build_scanner(self):
    terminals, self.callback = _create_unless(self.terminals, self.g_regex_flags, self.re, self.use_bytes)
    assert not self.user_callbacks
    for terminal_name, callback in self.callback.items():
        terminal = self.terminals_by_name[terminal_name]
        for sub_terminal in callback.scanner.terminals:
            self.terminals.remove(sub_terminal)
            idx = self.terminals.index(terminal)
            self.terminals.insert(idx, sub_terminal)
    self._scanner = PartialScanner(self.terminals, self.g_regex_flags, self.re, self.use_bytes)