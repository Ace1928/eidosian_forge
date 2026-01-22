from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Protocol, Tuple, Union
import interegular
from lark import Lark
from outlines import grammars
from outlines.caching import cache
from outlines.fsm.regex import create_fsm_index_tokenizer, make_deterministic_fsm
class CFGGuide(Guide):
    """Guide to generate text that is in the language of a context-free grammar."""

    def __init__(self, cfg_string: str, tokenizer):
        self.cfg_string = cfg_string
        self.tokenizer = tokenizer
        self.parser = Lark(cfg_string, parser='lalr', lexer='contextual', propagate_positions=False, maybe_placeholders=False, regex=True, import_paths=[grammars.GRAMMAR_PATH])
        self.terminal_regexps = dict()
        for terminal in self.parser.terminals:
            if terminal.pattern is not None:
                self.terminal_regexps[terminal.name] = terminal.pattern.to_regexp()
        self.terminal_regexps['$END'] = tokenizer.eos_token
        self.generation = ''
        self.reset_state = False
        self.allow_eos = False
        self.regex_fsm: RegexGuide
        self.check_last = False
        self.proposal_last: List[int] = []
        self.regex_fsm_last: RegexGuide
        self.start_state = 0
        self.final_state = -1

    def get_next_instruction(self, state: int) -> Instruction:
        """Generate an instruction for the next step.

        Upon initialization, the CFG incremental parser is used to determine the
        first regex and construct the first FSM to generate the first terminal.

        This FSM is used for proposals until either:

        - The FSM is exhausted, and its only remaining option is the EOS token,
          in which case we feed the generated terminal to the
          CFG incremental parser and allow it to propose the next regex
          corresponding to the next set of valid terminals.
        - The current FSM can be exhausted, but the EOS token is not the only
          remaining option. In this case we allow proposal of current terminal
          extensions, store the current FSM and its state, then also use the CFG
          parser to propose a new regex corresponding to terminating the current
          terminal and starting the next one. The model can then sample from
          either of these sets to determine whether to extend the current
          terminal or terminate it and start the next one.

        The CFG incremental parser is allowed to propose the EOS token from any accepting state,
        and once it is generated, the FSM will continue to always generate the EOS token.

        Parameters
        ----------
        state
            The current state of the FSM.

        Returns
        -------
        A list that contains the tokens to mask.

        """
        if self.is_final_state(state):
            return Write([self.tokenizer.eos_token_id])
        proposal: List[int] = []
        if self.generation != '':
            if self.check_last:
                proposer = self.regex_fsm_last
            else:
                proposer = self.regex_fsm
            instruction = proposer.get_next_instruction(state)
            if isinstance(instruction, Write):
                proposal += instruction.tokens
            else:
                proposal += instruction.tokens
            if self.tokenizer.eos_token_id not in proposal:
                return Generate(proposal)
            self.check_last = False
            proposal = [x for x in proposal if x != self.tokenizer.eos_token_id]
            if len(proposal) > 0:
                self.check_last = True
                self.proposal_last = proposal.copy()
                self.regex_fsm_last = proposer
        interactive = self.parser.parse_interactive(self.generation)
        interactive.exhaust_lexer()
        options = {self.terminal_regexps[x] for x in interactive.accepts()}
        options |= {self.terminal_regexps[x] for x in self.parser.lexer_conf.ignore}
        if self.terminal_regexps['$END'] in options:
            options.remove(self.terminal_regexps['$END'])
            if len(options) == 0:
                return Write([self.tokenizer.eos_token_id])
            self.allow_eos = True
            options.add('')
            assert len(options) > 1
        regex_string = '(' + '|'.join(['(' + x + ')' for x in options]) + ')'
        self.regex_fsm = RegexGuide(regex_string, self.tokenizer)
        self.reset_state = True
        instruction = self.regex_fsm.get_next_instruction(self.start_state)
        if isinstance(instruction, Write):
            proposal += instruction.tokens
        else:
            proposal += instruction.tokens
        if self.allow_eos:
            self.allow_eos = False
        else:
            proposal = [x for x in proposal if x != self.tokenizer.eos_token_id]
            assert len(proposal) > 0
        return Generate(proposal)

    def get_next_state(self, state: int, token_id: int) -> int:
        """Update the state of the guide.

        Transitions the underlying regex FSM to its next state.
        If at max tokens or EOS token, transition permanently to the final state.
        Update stored partial generations for subsequent incremental parsing.

        Parameters
        ----------
        state
            The current state of the FSM.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The new state of the FSM.
        """
        if token_id == self.tokenizer.eos_token_id or state == self.final_state:
            return self.final_state
        self.generation += self.tokenizer.decode([token_id])[0]
        if self.check_last:
            if token_id in self.proposal_last:
                return self.regex_fsm_last.get_next_state(state, token_id)
            self.check_last = False
        if self.reset_state:
            self.reset_state = False
            state = self.start_state
        return self.regex_fsm.get_next_state(state, token_id)

    def is_final_state(self, state: int) -> bool:
        return state == self.final_state

    def copy(self) -> 'CFGGuide':
        """Create a copy of the FSM."""
        return CFGGuide(self.cfg_string, self.tokenizer)