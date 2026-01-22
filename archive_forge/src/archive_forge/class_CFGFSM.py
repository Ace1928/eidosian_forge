import warnings
from typing import TYPE_CHECKING, List, NewType
from outlines.fsm.guide import CFGGuide, RegexGuide, StopAtEOSGuide
class CFGFSM(CFGGuide):
    """FSM to generate text that is in the language of a context-free grammar."""

    def __init__(self, cfg_string: str, tokenizer):
        warnings.warn(UserWarning('The `CFGFSM` interface is deprecated and will be removed on 2024-06-01. Please use `CFGGuide` instead.'))
        super().__init__(cfg_string, tokenizer)

    def allowed_token_ids(self, state: FSMState) -> List[int]:
        return self.get_next_instruction(state).tokens

    def next_state(self, state: FSMState, token_id: int) -> FSMState:
        return FSMState(self.get_next_state(state, token_id))

    def copy(self) -> 'CFGFSM':
        """Create a copy of the FSM."""
        return CFGFSM(self.cfg_string, self.tokenizer)