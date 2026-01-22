import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
class ViewList:
    """
    List with extended functionality: slices of ViewList objects are child
    lists, linked to their parents. Changes made to a child list also affect
    the parent list.  A child list is effectively a "view" (in the SQL sense)
    of the parent list.  Changes to parent lists, however, do *not* affect
    active child lists.  If a parent list is changed, any active child lists
    should be recreated.

    The start and end of the slice can be trimmed using the `trim_start()` and
    `trim_end()` methods, without affecting the parent list.  The link between
    child and parent lists can be broken by calling `disconnect()` on the
    child list.

    Also, ViewList objects keep track of the source & offset of each item.
    This information is accessible via the `source()`, `offset()`, and
    `info()` methods.
    """

    def __init__(self, initlist=None, source=None, items=None, parent=None, parent_offset=None):
        self.data = []
        'The actual list of data, flattened from various sources.'
        self.items = []
        'A list of (source, offset) pairs, same length as `self.data`: the\n        source of each line and the offset of each line from the beginning of\n        its source.'
        self.parent = parent
        'The parent list.'
        self.parent_offset = parent_offset
        'Offset of this list from the beginning of the parent list.'
        if isinstance(initlist, ViewList):
            self.data = initlist.data[:]
            self.items = initlist.items[:]
        elif initlist is not None:
            self.data = list(initlist)
            if items:
                self.items = items
            else:
                self.items = [(source, i) for i in range(len(initlist))]
        assert len(self.data) == len(self.items), 'data mismatch'

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return '%s(%s, items=%s)' % (self.__class__.__name__, self.data, self.items)

    def __lt__(self, other):
        return self.data < self.__cast(other)

    def __le__(self, other):
        return self.data <= self.__cast(other)

    def __eq__(self, other):
        return self.data == self.__cast(other)

    def __ne__(self, other):
        return self.data != self.__cast(other)

    def __gt__(self, other):
        return self.data > self.__cast(other)

    def __ge__(self, other):
        return self.data >= self.__cast(other)

    def __cmp__(self, other):
        return cmp(self.data, self.__cast(other))

    def __cast(self, other):
        if isinstance(other, ViewList):
            return other.data
        else:
            return other

    def __contains__(self, item):
        return item in self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if isinstance(i, slice):
            assert i.step in (None, 1), 'cannot handle slice with stride'
            return self.__class__(self.data[i.start:i.stop], items=self.items[i.start:i.stop], parent=self, parent_offset=i.start or 0)
        else:
            return self.data[i]

    def __setitem__(self, i, item):
        if isinstance(i, slice):
            assert i.step in (None, 1), 'cannot handle slice with stride'
            if not isinstance(item, ViewList):
                raise TypeError('assigning non-ViewList to ViewList slice')
            self.data[i.start:i.stop] = item.data
            self.items[i.start:i.stop] = item.items
            assert len(self.data) == len(self.items), 'data mismatch'
            if self.parent:
                self.parent[(i.start or 0) + self.parent_offset:(i.stop or len(self)) + self.parent_offset] = item
        else:
            self.data[i] = item
            if self.parent:
                self.parent[i + self.parent_offset] = item

    def __delitem__(self, i):
        try:
            del self.data[i]
            del self.items[i]
            if self.parent:
                del self.parent[i + self.parent_offset]
        except TypeError:
            assert i.step is None, 'cannot handle slice with stride'
            del self.data[i.start:i.stop]
            del self.items[i.start:i.stop]
            if self.parent:
                del self.parent[(i.start or 0) + self.parent_offset:(i.stop or len(self)) + self.parent_offset]

    def __add__(self, other):
        if isinstance(other, ViewList):
            return self.__class__(self.data + other.data, items=self.items + other.items)
        else:
            raise TypeError('adding non-ViewList to a ViewList')

    def __radd__(self, other):
        if isinstance(other, ViewList):
            return self.__class__(other.data + self.data, items=other.items + self.items)
        else:
            raise TypeError('adding ViewList to a non-ViewList')

    def __iadd__(self, other):
        if isinstance(other, ViewList):
            self.data += other.data
        else:
            raise TypeError('argument to += must be a ViewList')
        return self

    def __mul__(self, n):
        return self.__class__(self.data * n, items=self.items * n)
    __rmul__ = __mul__

    def __imul__(self, n):
        self.data *= n
        self.items *= n
        return self

    def extend(self, other):
        if not isinstance(other, ViewList):
            raise TypeError('extending a ViewList with a non-ViewList')
        if self.parent:
            self.parent.insert(len(self.data) + self.parent_offset, other)
        self.data.extend(other.data)
        self.items.extend(other.items)

    def append(self, item, source=None, offset=0):
        if source is None:
            self.extend(item)
        else:
            if self.parent:
                self.parent.insert(len(self.data) + self.parent_offset, item, source, offset)
            self.data.append(item)
            self.items.append((source, offset))

    def insert(self, i, item, source=None, offset=0):
        if source is None:
            if not isinstance(item, ViewList):
                raise TypeError('inserting non-ViewList with no source given')
            self.data[i:i] = item.data
            self.items[i:i] = item.items
            if self.parent:
                index = (len(self.data) + i) % len(self.data)
                self.parent.insert(index + self.parent_offset, item)
        else:
            self.data.insert(i, item)
            self.items.insert(i, (source, offset))
            if self.parent:
                index = (len(self.data) + i) % len(self.data)
                self.parent.insert(index + self.parent_offset, item, source, offset)

    def pop(self, i=-1):
        if self.parent:
            index = (len(self.data) + i) % len(self.data)
            self.parent.pop(index + self.parent_offset)
        self.items.pop(i)
        return self.data.pop(i)

    def trim_start(self, n=1):
        """
        Remove items from the start of the list, without touching the parent.
        """
        if n > len(self.data):
            raise IndexError("Size of trim too large; can't trim %s items from a list of size %s." % (n, len(self.data)))
        elif n < 0:
            raise IndexError('Trim size must be >= 0.')
        del self.data[:n]
        del self.items[:n]
        if self.parent:
            self.parent_offset += n

    def trim_end(self, n=1):
        """
        Remove items from the end of the list, without touching the parent.
        """
        if n > len(self.data):
            raise IndexError("Size of trim too large; can't trim %s items from a list of size %s." % (n, len(self.data)))
        elif n < 0:
            raise IndexError('Trim size must be >= 0.')
        del self.data[-n:]
        del self.items[-n:]

    def remove(self, item):
        index = self.index(item)
        del self[index]

    def count(self, item):
        return self.data.count(item)

    def index(self, item):
        return self.data.index(item)

    def reverse(self):
        self.data.reverse()
        self.items.reverse()
        self.parent = None

    def sort(self, *args):
        tmp = list(zip(self.data, self.items))
        tmp.sort(*args)
        self.data = [entry[0] for entry in tmp]
        self.items = [entry[1] for entry in tmp]
        self.parent = None

    def info(self, i):
        """Return source & offset for index `i`."""
        try:
            return self.items[i]
        except IndexError:
            if i == len(self.data):
                return (self.items[i - 1][0], None)
            else:
                raise

    def source(self, i):
        """Return source for index `i`."""
        return self.info(i)[0]

    def offset(self, i):
        """Return offset for index `i`."""
        return self.info(i)[1]

    def disconnect(self):
        """Break link between this list and parent list."""
        self.parent = None

    def xitems(self):
        """Return iterator yielding (source, offset, value) tuples."""
        for value, (source, offset) in zip(self.data, self.items):
            yield (source, offset, value)

    def pprint(self):
        """Print the list in `grep` format (`source:offset:value` lines)"""
        for line in self.xitems():
            print('%s:%d:%s' % line)