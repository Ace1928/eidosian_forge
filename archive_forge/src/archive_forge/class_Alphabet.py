from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
class Alphabet(Mapping[Any, TransitionKey]):

    @property
    def by_transition(self):
        return self._by_transition

    def __str__(self):
        out = []
        width = 0
        for tk, symbols in sorted(self._by_transition.items()):
            out.append((nice_char_group(symbols), str(tk)))
            if len(out[-1][0]) > width:
                width = len(out[-1][0])
        return '\n'.join((f'{a:{width}} | {b}' for a, b in out))

    def __repr__(self):
        return f'{type(self).__name__}({self._symbol_mapping!r})'

    def __len__(self) -> int:
        return len(self._symbol_mapping)

    def __iter__(self):
        return iter(self._symbol_mapping)

    def __init__(self, symbol_mapping: Dict[Union[str, _AnythingElseCls], TransitionKey]):
        self._symbol_mapping = symbol_mapping
        by_transition = defaultdict(list)
        for s, t in self._symbol_mapping.items():
            by_transition[t].append(s)
        self._by_transition = dict(by_transition)

    def __getitem__(self, item):
        if item not in self._symbol_mapping:
            if anything_else in self._symbol_mapping:
                return self._symbol_mapping[anything_else]
            else:
                return None
        else:
            return self._symbol_mapping[item]

    def __contains__(self, item):
        return item in self._symbol_mapping

    def union(*alphabets: 'Alphabet') -> 'Tuple[Alphabet, Tuple[Dict[TransitionKey, TransitionKey], ...]]':
        all_symbols = frozenset().union(*(a._symbol_mapping.keys() for a in alphabets))
        symbol_to_keys = {symbol: tuple((a[symbol] for a in alphabets)) for symbol in all_symbols}
        keys_to_symbols = defaultdict(list)
        for symbol, keys in symbol_to_keys.items():
            keys_to_symbols[keys].append(symbol)
        keys_to_key = {k: i for i, k in enumerate(keys_to_symbols)}
        result = Alphabet({symbol: keys_to_key[keys] for keys, symbols in keys_to_symbols.items() for symbol in symbols})
        new_to_old_mappings = [{} for _ in alphabets]
        for keys, new_key in keys_to_key.items():
            for old_key, new_to_old in zip(keys, new_to_old_mappings):
                new_to_old[new_key] = old_key
        return (result, tuple(new_to_old_mappings))

    @classmethod
    def from_groups(cls, *groups):
        return Alphabet({s: TransitionKey(i) for i, group in enumerate(groups) for s in group})

    def intersect(self, other: 'Alphabet') -> 'Tuple[Alphabet, Tuple[Dict[TransitionKey, TransitionKey], ...]]':
        all_symbols = frozenset(self._symbol_mapping).intersection(other._symbol_mapping)
        symbol_to_keys = {symbol: tuple((a[symbol] for a in (self, other))) for symbol in all_symbols}
        keys_to_symbols = defaultdict(list)
        for symbol, keys in symbol_to_keys.items():
            keys_to_symbols[keys].append(symbol)
        keys_to_key = {k: i for i, k in enumerate(keys_to_symbols)}
        result = Alphabet({symbol: keys_to_key[keys] for keys, symbols in keys_to_symbols.items() for symbol in symbols})
        old_to_new_mappings = [defaultdict(list) for _ in (self, other)]
        new_to_old_mappings = [{} for _ in (self, other)]
        for keys, new_key in keys_to_key.items():
            for old_key, old_to_new, new_to_old in zip(keys, old_to_new_mappings, new_to_old_mappings):
                old_to_new[old_key].append(new_key)
                new_to_old[new_key] = old_key
        return (result, tuple(new_to_old_mappings))

    def copy(self):
        return Alphabet(self._symbol_mapping.copy())