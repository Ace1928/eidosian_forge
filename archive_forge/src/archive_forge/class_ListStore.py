import warnings
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..overrides import override, deprecated_init, wrap_list_store_sort_func
from ..module import get_introspection_module
from gi import PyGIWarning
from gi.repository import GLib
import sys
class ListStore(Gio.ListStore):

    def sort(self, compare_func, *user_data):
        compare_func = wrap_list_store_sort_func(compare_func)
        return super(ListStore, self).sort(compare_func, *user_data)

    def insert_sorted(self, item, compare_func, *user_data):
        compare_func = wrap_list_store_sort_func(compare_func)
        return super(ListStore, self).insert_sorted(item, compare_func, *user_data)

    def __delitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step == 1:
                _list_store_splice(self, start, max(stop - start, 0), [])
            elif step == -1:
                _list_store_splice(self, stop + 1, max(start - stop, 0), [])
            else:
                for i in sorted(range(start, stop, step), reverse=True):
                    self.remove(i)
        elif isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError
            self.remove(key)
        else:
            raise TypeError

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            pytype = self.get_item_type().pytype
            valuelist = []
            for v in value:
                if not isinstance(v, pytype):
                    raise TypeError('Expected type %s.%s' % (pytype.__module__, pytype.__name__))
                valuelist.append(v)
            start, stop, step = key.indices(len(self))
            if step == 1:
                _list_store_splice(self, start, max(stop - start, 0), valuelist)
            else:
                indices = list(range(start, stop, step))
                if len(indices) != len(valuelist):
                    raise ValueError
                if step == -1:
                    _list_store_splice(self, stop + 1, max(start - stop, 0), valuelist[::-1])
                else:
                    for i, v in zip(indices, valuelist):
                        _list_store_splice(self, i, 1, [v])
        elif isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError
            pytype = self.get_item_type().pytype
            if not isinstance(value, pytype):
                raise TypeError('Expected type %s.%s' % (pytype.__module__, pytype.__name__))
            _list_store_splice(self, key, 1, [value])
        else:
            raise TypeError