import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class TreeModel(Gtk.TreeModel):

    def __len__(self):
        return self.iter_n_children(None)

    def __bool__(self):
        return True
    if GTK3:
        __nonzero__ = __bool__

    def _getiter(self, key):
        if isinstance(key, Gtk.TreeIter):
            return key
        elif isinstance(key, int) and key < 0:
            index = len(self) + key
            if index < 0:
                raise IndexError('row index is out of bounds: %d' % key)
            return self.get_iter(index)
        else:
            try:
                aiter = self.get_iter(key)
            except ValueError:
                raise IndexError("could not find tree path '%s'" % key)
            return aiter

    def sort_new_with_model(self):
        super_object = super(TreeModel, self)
        if hasattr(super_object, 'sort_new_with_model'):
            return super_object.sort_new_with_model()
        else:
            return TreeModelSort.new_with_model(self)

    def _coerce_path(self, path):
        if isinstance(path, Gtk.TreePath):
            return path
        else:
            return TreePath(path)

    def __getitem__(self, key):
        aiter = self._getiter(key)
        return TreeModelRow(self, aiter)

    def __setitem__(self, key, value):
        row = self[key]
        self.set_row(row.iter, value)

    def __delitem__(self, key):
        aiter = self._getiter(key)
        self.remove(aiter)

    def __iter__(self):
        return TreeModelRowIter(self, self.get_iter_first())
    get_iter_first = strip_boolean_result(Gtk.TreeModel.get_iter_first)
    iter_children = strip_boolean_result(Gtk.TreeModel.iter_children)
    iter_nth_child = strip_boolean_result(Gtk.TreeModel.iter_nth_child)
    iter_parent = strip_boolean_result(Gtk.TreeModel.iter_parent)
    get_iter_from_string = strip_boolean_result(Gtk.TreeModel.get_iter_from_string, ValueError, 'invalid tree path')

    def get_iter(self, path):
        path = self._coerce_path(path)
        success, aiter = super(TreeModel, self).get_iter(path)
        if not success:
            raise ValueError("invalid tree path '%s'" % path)
        return aiter

    def iter_next(self, aiter):
        next_iter = aiter.copy()
        success = super(TreeModel, self).iter_next(next_iter)
        if success:
            return next_iter

    def iter_previous(self, aiter):
        prev_iter = aiter.copy()
        success = super(TreeModel, self).iter_previous(prev_iter)
        if success:
            return prev_iter

    def _convert_row(self, row):
        if isinstance(row, str):
            raise TypeError('Expected a list or tuple, but got str')
        n_columns = self.get_n_columns()
        if len(row) != n_columns:
            raise ValueError('row sequence has the incorrect number of elements')
        result = []
        columns = []
        for cur_col, value in enumerate(row):
            if value is None:
                continue
            result.append(self._convert_value(cur_col, value))
            columns.append(cur_col)
        return (result, columns)

    def set_row(self, treeiter, row):
        converted_row, columns = self._convert_row(row)
        for column in columns:
            self.set_value(treeiter, column, row[column])

    def _convert_value(self, column, value):
        """Convert value to a GObject.Value of the expected type"""
        if isinstance(value, GObject.Value):
            return value
        return GObject.Value(self.get_column_type(column), value)

    def get(self, treeiter, *columns):
        n_columns = self.get_n_columns()
        values = []
        for col in columns:
            if not isinstance(col, int):
                raise TypeError('column numbers must be ints')
            if col < 0 or col >= n_columns:
                raise ValueError('column number is out of range')
            values.append(self.get_value(treeiter, col))
        return tuple(values)

    def row_changed(self, path, iter):
        return super(TreeModel, self).row_changed(self._coerce_path(path), iter)

    def row_inserted(self, path, iter):
        return super(TreeModel, self).row_inserted(self._coerce_path(path), iter)

    def row_has_child_toggled(self, path, iter):
        return super(TreeModel, self).row_has_child_toggled(self._coerce_path(path), iter)

    def row_deleted(self, path):
        return super(TreeModel, self).row_deleted(self._coerce_path(path))

    def rows_reordered(self, path, iter, new_order):
        return super(TreeModel, self).rows_reordered(self._coerce_path(path), iter, new_order)