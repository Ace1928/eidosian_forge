import collections.abc
class ImmutableDict(collections.abc.Mapping):
    """Immutable `Mapping`."""

    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return f'ImmutableDict({self._dict})'
    __supported_by_tf_nest__ = True