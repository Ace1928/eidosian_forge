import operator
import typing
from collections.abc import MappingView, MutableMapping, MutableSet
class ParameterReferences(MutableSet):
    """A set of instruction parameter slot references.
    Items are expected in the form ``(instruction, param_index)``. Membership
    testing is overridden such that items that are otherwise value-wise equal
    are still considered distinct if their ``instruction``\\ s are referentially
    distinct.

    In the case of the special value :attr:`.ParameterTable.GLOBAL_PHASE` for ``instruction``, the
    ``param_index`` should be ``None``.
    """

    def _instance_key(self, ref):
        return (id(ref[0]), ref[1])

    def __init__(self, refs):
        self._instance_ids = {}
        for ref in refs:
            if not isinstance(ref, tuple) or len(ref) != 2:
                raise ValueError('refs must be in form (instruction, param_index)')
            k = self._instance_key(ref)
            self._instance_ids[k] = ref[0]

    def __getstate__(self):
        return list(self)

    def __setstate__(self, refs):
        self._instance_ids = {self._instance_key(ref): ref[0] for ref in refs}

    def __len__(self):
        return len(self._instance_ids)

    def __iter__(self):
        for (_, idx), instruction in self._instance_ids.items():
            yield (instruction, idx)

    def __contains__(self, x) -> bool:
        return self._instance_key(x) in self._instance_ids

    def __repr__(self) -> str:
        return f'ParameterReferences({repr(list(self))})'

    def add(self, value):
        """Adds a reference to the listing if it's not already present."""
        k = self._instance_key(value)
        self._instance_ids[k] = value[0]

    def discard(self, value):
        k = self._instance_key(value)
        self._instance_ids.pop(k, None)

    def copy(self):
        """Create a shallow copy."""
        return ParameterReferences(self)