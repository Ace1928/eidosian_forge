from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Protocol, Tuple, Union
import interegular
from lark import Lark
from outlines import grammars
from outlines.caching import cache
from outlines.fsm.regex import create_fsm_index_tokenizer, make_deterministic_fsm
class RegexGuide(Guide):
    """Guide to generate text in the language of a regular expression."""
    initial_state = 0

    def __init__(self, regex_string: str, tokenizer):

        @cache()
        def create_states_mapping(regex_string: str, cacheable_vocabulary: Tuple[Tuple[str, int], ...]) -> Tuple[dict, set, set]:
            """Create the variables related to the mapping between states and tokens
            The parameters of the function are used for caching purpose
            """
            regex_pattern = interegular.parse_pattern(regex_string)
            regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())
            states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(regex_fsm, tokenizer)
            if not any((regex_fsm.finals.intersection(v.values()) for v in states_to_token_maps.values())):
                raise ValueError('The vocabulary does not allow us to build a sequence that matches the input regex')
            return (states_to_token_maps, empty_token_ids, regex_fsm.finals)
        self.states_to_token_maps, self.empty_token_ids, fsm_finals = create_states_mapping(regex_string, tuple(sorted(tokenizer.vocabulary.items())))
        self.vocabulary = list(tokenizer.vocabulary.values())
        self.eos_token_id = tokenizer.eos_token_id
        self.final_states = fsm_finals | {-1}

    def get_next_instruction(self, state: int) -> Instruction:
        """Return the next instruction for guided generation.

        The initialization of the guide builds an index which maps FSM states to a
        map from authorized tokens to the state in which the guide needs to move
        if said token is generated. Therefore the authorized tokens at the
        current state are the keys of the map returned by the value of the index
        for current state.

        If the current state is not contained in the end this means that we are
        in a final state of the guide. We only authorize EOS tokens in the final
        state.

        Parameters
        ----------
        state
            The current state of the guide.

        Returns
        -------
        A `Generate` instance that contains the model and the allowed token ids.

        """
        next_tokens_to_end_states = self.states_to_token_maps.get(state)
        if next_tokens_to_end_states is None:
            return Write([self.eos_token_id])
        return Generate(list(next_tokens_to_end_states.keys()))

    def get_next_state(self, state: int, token_id: int) -> int:
        """Update the state of the guide.

        We use the index to determine to which state the guide should transition
        given the token that was just generated.

        Parameters
        ----------
        state
            The current state of the guide.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The new state of the guide.

        """
        if token_id == self.eos_token_id:
            return -1
        elif state in self.final_states:
            return state
        last_token_to_end_state = self.states_to_token_maps[state]
        next_state = last_token_to_end_state.get(token_id)
        if next_state is None:
            next_state = -1
        return next_state

    @classmethod
    def from_interegular_fsm(cls, interegular_fsm: interegular.fsm.FSM, tokenizer: 'Tokenizer'):
        from_interegular_instance = cls.__new__(cls)

        def create_states_mapping_from_interegular_fsm(fsm: interegular.fsm.FSM, cacheable_vocabulary: Tuple[Tuple[str, int], ...]) -> Tuple[dict, set]:
            """Create the variables related to the mapping between states and tokens
            The parameters of the function are used for caching purpose
            """
            regex_fsm, _ = make_deterministic_fsm(fsm.reduce())
            states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(regex_fsm, tokenizer)
            if not any((regex_fsm.finals.intersection(v.values()) for v in states_to_token_maps.values())):
                raise ValueError('The vocabulary does not allow us to build a sequence that matches the input regex')
            return (states_to_token_maps, empty_token_ids)
        from_interegular_instance.states_to_token_maps, from_interegular_instance.empty_token_ids = create_states_mapping_from_interegular_fsm(interegular_fsm, tuple(sorted(tokenizer.vocabulary.items())))
        from_interegular_instance.vocabulary = list(tokenizer.vocabulary.values())
        from_interegular_instance.eos_token_id = tokenizer.eos_token_id
        return from_interegular_instance

    def is_final_state(self, state: int) -> bool:
        """Determine whether the current state of the guide is a final state."""
        return state in self.final_states

    def copy(self):
        return self