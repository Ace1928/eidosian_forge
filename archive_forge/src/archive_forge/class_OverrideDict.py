import warnings
from typing import (
from torch.onnx import _constants, errors
from torch.onnx._internal import _beartype
class OverrideDict(Generic[_K, _V], Collection[_K]):
    """A dictionary that merges built-in and custom symbolic functions.

    It supports overriding and un-overriding built-in symbolic functions with custom
    ones.
    """

    def __init__(self):
        self._base: Dict[_K, _V] = {}
        self._overrides: Dict[_K, _V] = {}
        self._merged: Dict[_K, _V] = {}

    def set_base(self, key: _K, value: _V) -> None:
        self._base[key] = value
        if key not in self._overrides:
            self._merged[key] = value

    def in_base(self, key: _K) -> bool:
        """Checks if a key is in the base dictionary."""
        return key in self._base

    def override(self, key: _K, value: _V) -> None:
        """Overrides a base key-value with a new pair."""
        self._overrides[key] = value
        self._merged[key] = value

    def remove_override(self, key: _K) -> None:
        """Un-overrides a key-value pair."""
        self._overrides.pop(key, None)
        self._merged.pop(key, None)
        if key in self._base:
            self._merged[key] = self._base[key]

    def overridden(self, key: _K) -> bool:
        """Checks if a key-value pair is overridden."""
        return key in self._overrides

    def __getitem__(self, key: _K) -> _V:
        return self._merged[key]

    def get(self, key: _K, default: Optional[_V]=None):
        return self._merged.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._merged

    def __iter__(self):
        return iter(self._merged)

    def __len__(self) -> int:
        return len(self._merged)

    def __repr__(self) -> str:
        return f'OverrideDict(base={self._base}, overrides={self._overrides})'

    def __bool__(self) -> bool:
        return bool(self._merged)