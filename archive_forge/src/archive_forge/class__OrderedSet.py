class _OrderedSet(object):

    def __init__(self, initial_contents=None):
        self._contents = []
        self._contents_as_set = set()
        if initial_contents is not None:
            for x in initial_contents:
                self.add(x)

    def add(self, x):
        if x not in self._contents_as_set:
            self._contents_as_set.add(x)
            self._contents.append(x)

    def discard(self, x):
        if x in self._contents_as_set:
            self._contents_as_set.remove(x)
            self._contents.remove(x)

    def copy(self):
        return _OrderedSet(self._contents)

    def update(self, contents):
        for x in contents:
            self.add(x)

    def __iter__(self):
        return iter(self._contents)

    def __contains__(self, item):
        return item in self._contents_as_set

    def __len__(self):
        return len(self._contents)

    def set_repr(self):
        if len(self) == 0:
            return 'set()'
        lst = [repr(x) for x in self]
        return 'set([' + ', '.join(lst) + '])'