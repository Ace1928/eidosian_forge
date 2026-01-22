import collections.abc
import functools
import itertools
import threading
import typing as ty
import uuid
import warnings
import debtcollector
from debtcollector import renames
class _DeprecatedPolicyValues(collections.abc.MutableMapping):
    """A Dictionary that manages current and deprecated policy values.

    Anything added to this dictionary after initial creation is considered a
    deprecated key that we are trying to move services away from. Accessing
    these values as oslo.policy will do will trigger a DeprecationWarning.
    """

    def __init__(self, data: ty.Dict[str, ty.Any]):
        self._data = data
        self._deprecated: ty.Dict[str, ty.Any] = {}

    def __getitem__(self, k: str) -> ty.Any:
        try:
            return self._data[k]
        except KeyError:
            pass
        try:
            val = self._deprecated[k]
        except KeyError:
            pass
        else:
            warnings.warn('Policy enforcement is depending on the value of %s. This key is deprecated. Please update your policy file to use the standard policy values.' % k, DeprecationWarning)
            return val
        raise KeyError(k)

    def __setitem__(self, k: str, v: ty.Any) -> None:
        self._deprecated[k] = v

    def __delitem__(self, k: str) -> None:
        del self._deprecated[k]

    def __iter__(self) -> ty.Iterator[ty.Any]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __str__(self) -> str:
        return self._dict.__str__()

    def __repr__(self) -> str:
        return self._dict.__repr__()

    @property
    def _dict(self) -> ty.Dict[str, ty.Any]:
        d = self._deprecated.copy()
        d.update(self._data)
        return d