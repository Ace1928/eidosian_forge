from kivy.properties import ListProperty, ObservableDict, ObjectProperty
from kivy.event import EventDispatcher
from functools import partial
class RecycleDataModel(RecycleDataModelBehavior, EventDispatcher):
    """An implementation of :class:`RecycleDataModelBehavior` that keeps the
    data in a indexable list. See :attr:`data`.

    When data changes this class currently dispatches `on_data_changed`  with
    one of the following additional keyword arguments.

    `none`: no keyword argument
        With no additional argument it means a generic data change.
    `removed`: a slice or integer
        The value is a slice or integer indicating the indices removed.
    `appended`: a slice
        The slice in :attr:`data` indicating the first and last new items
        (i.e. the slice pointing to the new items added at the end).
    `inserted`: a integer
        The index in :attr:`data` where a new data item was inserted.
    `modified`: a slice
        The slice with the indices where the data has been modified.
        This currently does not allow changing of size etc.
    """
    data = ListProperty([])
    "Stores the model's data using a list.\n\n    The data for a item at index `i` can also be accessed with\n    :class:`RecycleDataModel` `[i]`.\n    "
    _last_len = 0

    def __init__(self, **kwargs):
        self.fbind('data', self._on_data_callback)
        super(RecycleDataModel, self).__init__(**kwargs)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def observable_dict(self):
        """A dictionary instance, which when modified will trigger a `data` and
        consequently an `on_data_changed` dispatch.
        """
        return partial(ObservableDict, self.__class__.data, self)

    def attach_recycleview(self, rv):
        super(RecycleDataModel, self).attach_recycleview(rv)
        if rv:
            self.fbind('data', rv._dispatch_prop_on_source, 'data')

    def detach_recycleview(self):
        rv = self.recycleview
        if rv:
            self.funbind('data', rv._dispatch_prop_on_source, 'data')
        super(RecycleDataModel, self).detach_recycleview()

    def _on_data_callback(self, instance, value):
        last_len = self._last_len
        new_len = self._last_len = len(self.data)
        op, val = value.last_op
        if op == '__setitem__':
            val = recondition_slice_assign(val, last_len, new_len)
            if val is not None:
                self.dispatch('on_data_changed', modified=val)
            else:
                self.dispatch('on_data_changed')
        elif op == '__delitem__':
            self.dispatch('on_data_changed', removed=val)
        elif op == '__setslice__':
            val = recondition_slice_assign(slice(*val), last_len, new_len)
            if val is not None:
                self.dispatch('on_data_changed', modified=val)
            else:
                self.dispatch('on_data_changed')
        elif op == '__delslice__':
            self.dispatch('on_data_changed', removed=slice(*val))
        elif op == '__iadd__' or op == '__imul__':
            self.dispatch('on_data_changed', appended=slice(last_len, new_len))
        elif op == 'append':
            self.dispatch('on_data_changed', appended=slice(last_len, new_len))
        elif op == 'insert':
            self.dispatch('on_data_changed', inserted=val)
        elif op == 'pop':
            if val:
                self.dispatch('on_data_changed', removed=val[0])
            else:
                self.dispatch('on_data_changed', removed=last_len - 1)
        elif op == 'extend':
            self.dispatch('on_data_changed', appended=slice(last_len, new_len))
        else:
            self.dispatch('on_data_changed')