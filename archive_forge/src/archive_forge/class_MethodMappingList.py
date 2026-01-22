import re
class MethodMappingList(list):
    """
    Like a list but allows calling a method on it means that it is called
    for all its elements.

    >>> a = MethodMappingList([2+1j, 3+2j, 4+2j])
    >>> a.conjugate()
    [(2-1j), (3-2j), (4-2j)]

    This can be nested:

    >>> b = MethodMappingList([1+1j, a])
    >>> b.conjugate()
    [(1-1j), [(2-1j), (3-2j), (4-2j)]]

    Also supports flattening:

    >>> b.flatten()
    [(1+1j), (2+1j), (3+2j), (4+2j)]

    """

    def __init__(self, l=[], p=None):
        super(MethodMappingList, self).__init__(l)

    def __call__(self, *args, **kwargs):
        return type(self)([elt(*args, **kwargs) for elt in self], p=self)

    def __getattr__(self, attr):
        return type(self)([getattr(e, attr) for e in self], p=self)

    def flatten(self, depth=1):
        return _flatten(self, depth=depth)