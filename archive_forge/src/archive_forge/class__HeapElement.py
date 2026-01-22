import heapq
class _HeapElement:
    """This proxy class separates the heap element from its priority.

    The idea is that using a 2-tuple (priority, element) works
    for sorting, but not for dict lookup because priorities are
    often floating point values so round-off can mess up equality.

    So, we need inequalities to look at the priority (for sorting)
    and equality (and hash) to look at the element to enable
    updates to the priority.

    Unfortunately, this class can be tricky to work with if you forget that
    `__lt__` compares the priority while `__eq__` compares the element.
    In `greedy_modularity_communities()` the following code is
    used to check that two _HeapElements differ in either element or priority:

        if d_oldmax != row_max or d_oldmax.priority != row_max.priority:

    If the priorities are the same, this implementation uses the element
    as a tiebreaker. This provides compatibility with older systems that
    use tuples to combine priority and elements.
    """
    __slots__ = ['priority', 'element', '_hash']

    def __init__(self, priority, element):
        self.priority = priority
        self.element = element
        self._hash = hash(element)

    def __lt__(self, other):
        try:
            other_priority = other.priority
        except AttributeError:
            return self.priority < other
        if self.priority == other_priority:
            try:
                return self.element < other.element
            except TypeError as err:
                raise TypeError('Consider using a tuple, with a priority value that can be compared.')
        return self.priority < other_priority

    def __gt__(self, other):
        try:
            other_priority = other.priority
        except AttributeError:
            return self.priority > other
        if self.priority == other_priority:
            try:
                return self.element > other.element
            except TypeError as err:
                raise TypeError('Consider using a tuple, with a priority value that can be compared.')
        return self.priority > other_priority

    def __eq__(self, other):
        try:
            return self.element == other.element
        except AttributeError:
            return self.element == other

    def __hash__(self):
        return self._hash

    def __getitem__(self, indx):
        return self.priority if indx == 0 else self.element[indx - 1]

    def __iter__(self):
        yield self.priority
        try:
            yield from self.element
        except TypeError:
            yield self.element

    def __repr__(self):
        return f'_HeapElement({self.priority}, {self.element})'