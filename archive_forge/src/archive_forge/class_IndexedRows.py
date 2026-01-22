import collections
import functools
import operator
from ovs.db import data
class IndexedRows(DictBase, object):

    def __init__(self, table, *args, **kwargs):
        super(IndexedRows, self).__init__(*args, **kwargs)
        self.table = table
        self.indexes = {}
        self.IndexEntry = IndexEntryClass(table)

    def index_create(self, name):
        if name in self.indexes:
            raise ValueError('An index named {} already exists'.format(name))
        index = self.indexes[name] = MultiColumnIndex(name)
        return index

    def __setitem__(self, key, item):
        self.data[key] = item
        for index in self.indexes.values():
            index.add(item)

    def __delitem__(self, key):
        val = self.data[key]
        del self.data[key]
        for index in self.indexes.values():
            index.remove(val)

    def clear(self):
        self.data.clear()
        for index in self.indexes.values():
            index.clear()

    def update(self, dict=None, **kwargs):
        raise NotImplementedError()

    def setdefault(self, key, failobj=None):
        raise NotImplementedError()

    def pop(self, key, *args):
        raise NotImplementedError()

    def popitem(self):
        raise NotImplementedError()

    @classmethod
    def fromkeys(cls, iterable, value=None):
        raise NotImplementedError()