from __future__ import annotations
class EqualityDict(dict):
    """Dict using the eq operator for keying."""

    def __getitem__(self, key):
        h = eqhash(key)
        if h not in self:
            return self.__missing__(key)
        return super().__getitem__(h)

    def __setitem__(self, key, value):
        return super().__setitem__(eqhash(key), value)

    def __delitem__(self, key):
        return super().__delitem__(eqhash(key))