from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Protocol, Tuple, Union
import interegular
from lark import Lark
from outlines import grammars
from outlines.caching import cache
from outlines.fsm.regex import create_fsm_index_tokenizer, make_deterministic_fsm
class StopAtEOSGuide(Guide):
    """Guide to generate tokens until the EOS token has been generated."""
    final_state = 1
    start_state = 0

    def __init__(self, tokenizer: 'Tokenizer'):
        """Initialize the generation guide.

        model
            The logit generator used to generate the next token.

        """
        self.eos_token_id = tokenizer.eos_token_id
        self.vocabulary = tokenizer.vocabulary.values()

    def get_next_instruction(self, state: int) -> Instruction:
        if self.is_final_state(state):
            return Write([self.eos_token_id])
        return Generate(list(self.vocabulary))

    def get_next_state(self, state: int, token_id: int) -> int:
        if token_id == self.eos_token_id or state == self.final_state:
            return self.final_state
        return self.start_state

    def is_final_state(self, state: int):
        return state == self.final_state

    def copy(self):
        return self