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
class PartialScanner(Scanner):

    @classmethod
    @lru_cache
    def construct_terminal_fsm(cls, terminal):
        regex_str = terminal.pattern.to_regexp()
        pattern = interegular.parse_pattern(regex_str)
        fsm, _ = make_deterministic_fsm(pattern.to_fsm().reduce())
        return (fsm, pattern.prefix_postfix)

    def __init__(self, terminals, g_regex_flags, re_, use_bytes, match_whole=False):
        self.terminals = terminals
        self.g_regex_flags = g_regex_flags
        self.use_bytes = use_bytes
        self.match_whole = match_whole
        self.allowed_types = {t.name for t in self.terminals}
        self._mres = None
        fsms = []
        for t in self.terminals:
            fsm, prefix_postfix = self.construct_terminal_fsm(t)
            assert prefix_postfix == (0, 0)
            fsms.append(fsm)
        self.fsm, self.fsms_to_trans_finals = fsm_union(fsms)

    def get_terminals_info(self, fsm_state_seq) -> Tuple[Tuple[PartialTerminalInfo, ...], Tuple[PartialTerminalInfo, ...]]:
        """Get the possible terminal symbols for an FSM state sequence."""
        terminals_and_info: Tuple[PartialTerminalInfo, ...] = ()
        final_terminals_and_info: Tuple[PartialTerminalInfo, ...] = ()
        for i, (fsm_id, fsm_reads_more, in_final) in enumerate(get_sub_fsms_from_seq(fsm_state_seq, self.fsms_to_trans_finals)):
            terminal_name = self.terminals[fsm_id].name
            info = PartialTerminalInfo(i, terminal_name, fsm_reads_more, in_final)
            terminals_and_info += (info,)
            if in_final:
                final_terminals_and_info += (info,)
        return (terminals_and_info, final_terminals_and_info)

    def match(self, text, pos, last_fsm_state_seq: Optional[Tuple[int, ...]]=None):
        """Determine an FSM match over `text` starting at `pos` and continuing `last_fsm_state_seq`."""
        start_pos = pos
        if last_fsm_state_seq:
            assert len(last_fsm_state_seq) > 1
            start_pos += len(last_fsm_state_seq) - 1
            start_state = last_fsm_state_seq[-1]
        else:
            start_state = self.fsm.initial
        text_part = text[start_pos:]
        state_seq = walk_fsm(self.fsm, text_part, start_state, full_match=self.match_whole)
        if not state_seq:
            return None
        if last_fsm_state_seq:
            res = last_fsm_state_seq + tuple(state_seq)
        else:
            res = (start_state,) + tuple(state_seq)
        return res