class PairSet(object):
    __slots__ = ('_data',)

    def __init__(self):
        self._data = {}

    def __contains__(self, item):
        return self.has(item[0], item[1], item[2])

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return str(self._data)

    def has(self, a, b, are_mutually_exclusive):
        first = self._data.get(a)
        result = first and first.get(b)
        if result is None:
            return False
        if not are_mutually_exclusive:
            return not result
        return True

    def add(self, a, b, are_mutually_exclusive):
        _pair_set_add(self._data, a, b, are_mutually_exclusive)
        _pair_set_add(self._data, b, a, are_mutually_exclusive)
        return self