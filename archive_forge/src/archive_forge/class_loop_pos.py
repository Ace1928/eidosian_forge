import sys
from Cython.Tempita.compat3 import basestring_
class loop_pos(object):

    def __init__(self, seq, pos):
        self.seq = seq
        self.pos = pos

    def __repr__(self):
        return '<loop pos=%r at %r>' % (self.seq[self.pos], self.pos)

    def index(self):
        return self.pos
    index = property(index)

    def number(self):
        return self.pos + 1
    number = property(number)

    def item(self):
        return self.seq[self.pos]
    item = property(item)

    def __next__(self):
        try:
            return self.seq[self.pos + 1]
        except IndexError:
            return None
    __next__ = property(__next__)
    if sys.version < '3':
        next = __next__

    def previous(self):
        if self.pos == 0:
            return None
        return self.seq[self.pos - 1]
    previous = property(previous)

    def odd(self):
        return not self.pos % 2
    odd = property(odd)

    def even(self):
        return self.pos % 2
    even = property(even)

    def first(self):
        return self.pos == 0
    first = property(first)

    def last(self):
        return self.pos == len(self.seq) - 1
    last = property(last)

    def length(self):
        return len(self.seq)
    length = property(length)

    def first_group(self, getter=None):
        """
        Returns true if this item is the start of a new group,
        where groups mean that some attribute has changed.  The getter
        can be None (the item itself changes), an attribute name like
        ``'.attr'``, a function, or a dict key or list index.
        """
        if self.first:
            return True
        return self._compare_group(self.item, self.previous, getter)

    def last_group(self, getter=None):
        """
        Returns true if this item is the end of a new group,
        where groups mean that some attribute has changed.  The getter
        can be None (the item itself changes), an attribute name like
        ``'.attr'``, a function, or a dict key or list index.
        """
        if self.last:
            return True
        return self._compare_group(self.item, self.__next__, getter)

    def _compare_group(self, item, other, getter):
        if getter is None:
            return item != other
        elif isinstance(getter, basestring_) and getter.startswith('.'):
            getter = getter[1:]
            if getter.endswith('()'):
                getter = getter[:-2]
                return getattr(item, getter)() != getattr(other, getter)()
            else:
                return getattr(item, getter) != getattr(other, getter)
        elif hasattr(getter, '__call__'):
            return getter(item) != getter(other)
        else:
            return item[getter] != other[getter]