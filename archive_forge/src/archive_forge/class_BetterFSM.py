from collections import namedtuple
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Generator, List, Sequence, Set, Tuple
import numba
import numpy as np
from interegular.fsm import FSM, Alphabet, OblivionError, anything_else
from numba.typed.typedobjectutils import _nonoptional
class BetterFSM(FSM):
    flat_transition_map: Dict[Tuple[int, int], int]
    trans_key_to_states: Dict[int, List[int]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.alphabet, BetterAlphabet):
            self.__dict__['alphabet'] = BetterAlphabet(self.alphabet._symbol_mapping)
        flat_transition_map = {}
        trans_key_to_states = {}
        for from_state, trans_map in self.map.items():
            for trans_key, to_state in trans_map.items():
                flat_transition_map[from_state, trans_key] = to_state
                trans_key_to_states.setdefault(trans_key, set()).add(from_state)
        self.__dict__['trans_key_to_states'] = trans_key_to_states
        self.__dict__['flat_transition_map'] = flat_transition_map
        self.__dict__['_fsm_info'] = None

    def copy(self):
        return BetterFSM(alphabet=self.alphabet.copy(), states=self.states.copy(), initial=self.initial, finals=self.finals.copy(), map=self.map.copy(), __no_validation__=True)

    @property
    def fsm_info(self):
        if self._fsm_info is None:
            flat_transition_map_items = np.fromiter(((a[0], a[1], b) for a, b in self.flat_transition_map.items()), dtype=np.dtype('i8, i8, i8'))
            trans_key_to_states_items = np.fromiter(((k, z) for k, v in self.trans_key_to_states.items() for z in v), dtype=np.dtype('i8, i8'))
            alphabet_symbol_mapping_items = np.fromiter((it for it in self.alphabet._symbol_mapping.items() if it[0] != anything_else), dtype=np.dtype('U1, i8'))
            nb_finals = np.fromiter(self.finals, dtype=np.dtype('i8'))
            self.__dict__['_fsm_info'] = create_fsm_info(self.initial, nb_finals, flat_transition_map_items, trans_key_to_states_items, self.alphabet.anything_value, alphabet_symbol_mapping_items)
        return self._fsm_info