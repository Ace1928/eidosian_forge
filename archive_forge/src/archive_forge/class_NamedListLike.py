from __future__ import annotations
from collections import defaultdict, namedtuple
from typing import (
import param
from bokeh.models import Row as BkRow
from param.parameterized import iscoroutinefunction, resolve_ref
from ..io.document import freeze_doc
from ..io.model import hold
from ..io.resources import CDN_DIST
from ..io.state import state
from ..models import Column as PnColumn
from ..reactive import Reactive
from ..util import param_name, param_reprs, param_watchers
class NamedListLike(param.Parameterized):
    objects = param.List(default=[], doc='\n        The list of child objects that make up the layout.')
    _preprocess_params: ClassVar[List[str]] = ['objects']

    def __init__(self, *items: List[Any | Tuple[str, Any]], **params: Any):
        if 'objects' in params:
            if items:
                raise ValueError('%s objects should be supplied either as positional arguments or as a keyword, not both.' % type(self).__name__)
            items = params.pop('objects')
        params['objects'], self._names = self._to_objects_and_names(items)
        super().__init__(**params)
        self._panels = defaultdict(dict)
        self.param.watch(self._update_names, 'objects')
        param_watchers(self)['objects']['value'].reverse()

    def _to_object_and_name(self, item):
        from ..pane import panel
        if isinstance(item, tuple):
            name, item = item
        else:
            name = getattr(item, 'name', None)
        pane = panel(item, name=name)
        name = param_name(pane.name) if name is None else name
        return (pane, name)

    def _to_objects_and_names(self, items):
        objects, names = ([], [])
        for item in items:
            pane, name = self._to_object_and_name(item)
            objects.append(pane)
            names.append(name)
        return (objects, names)

    def _update_names(self, event: param.parameterized.Event) -> None:
        if len(event.new) == len(self._names):
            return
        names = []
        for obj in event.new:
            if obj in event.old:
                index = event.old.index(obj)
                name = self._names[index]
            else:
                name = obj.name
            names.append(name)
        self._names = names

    def _update_active(self, *events: param.parameterized.Event) -> None:
        pass

    def __getitem__(self, index) -> Viewable | List[Viewable]:
        return self.objects[index]

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self) -> Iterator[Viewable]:
        for obj in self.objects:
            yield obj

    def __iadd__(self, other: Iterable[Any]) -> 'NamedListLike':
        self.extend(other)
        return self

    def __add__(self, other: Iterable[Any]) -> 'NamedListLike':
        if isinstance(other, NamedListLike):
            added = list(zip(other._names, other.objects))
        elif isinstance(other, ListLike):
            added = other.objects
        else:
            added = list(other)
        objects = list(zip(self._names, self.objects))
        return self.clone(*objects + added)

    def __radd__(self, other: Iterable[Any]) -> 'NamedListLike':
        if isinstance(other, NamedListLike):
            added = list(zip(other._names, other.objects))
        elif isinstance(other, ListLike):
            added = other.objects
        else:
            added = list(other)
        objects = list(zip(self._names, self.objects))
        return self.clone(*added + objects)

    def __setitem__(self, index: int | slice, panes: Iterable[Any]) -> None:
        new_objects = list(self)
        if not isinstance(index, slice):
            if index > len(self.objects):
                raise IndexError('Index %d out of bounds on %s containing %d objects.' % (index, type(self).__name__, len(self.objects)))
            start, end = (index, index + 1)
            panes = [panes]
        else:
            start = index.start or 0
            end = len(self.objects) if index.stop is None else index.stop
            if index.start is None and index.stop is None:
                if not isinstance(panes, list):
                    raise IndexError('Expected a list of objects to replace the objects in the %s, got a %s type.' % (type(self).__name__, type(panes).__name__))
                expected = len(panes)
                new_objects = [None] * expected
                self._names = [None] * len(panes)
                end = expected
            else:
                expected = end - start
                if end > len(self.objects):
                    raise IndexError('Index %d out of bounds on %s containing %d objects.' % (end, type(self).__name__, len(self.objects)))
            if not isinstance(panes, list) or len(panes) != expected:
                raise IndexError('Expected a list of %d objects to set on the %s to match the supplied slice.' % (expected, type(self).__name__))
        for i, pane in zip(range(start, end), panes):
            new_objects[i], self._names[i] = self._to_object_and_name(pane)
        self.objects = new_objects

    def clone(self, *objects: Any, **params: Any) -> 'NamedListLike':
        """
        Makes a copy of the Tabs sharing the same parameters.

        Arguments
        ---------
        objects: Objects to add to the cloned Tabs object.
        params: Keyword arguments override the parameters on the clone.

        Returns
        -------
        Cloned Tabs object
        """
        if objects:
            overrides = objects
        elif 'objects' in params:
            raise ValueError('Tabs objects should be supplied either as positional arguments or as a keyword, not both.')
        elif 'objects' in params:
            overrides = params.pop('objects')
        else:
            overrides = tuple(zip(self._names, self.objects))
        p = dict(self.param.values(), **params)
        del p['objects']
        return type(self)(*overrides, **params)

    def append(self, pane: Any) -> None:
        """
        Appends an object to the tabs.

        Arguments
        ---------
        obj (object): Panel component to add as a tab.
        """
        new_object, new_name = self._to_object_and_name(pane)
        new_objects = list(self)
        new_objects.append(new_object)
        self._names.append(new_name)
        self.objects = new_objects

    def clear(self) -> None:
        """
        Clears the tabs.
        """
        self._names = []
        self.objects = []

    def extend(self, panes: Iterable[Any]) -> None:
        """
        Extends the the tabs with a list.

        Arguments
        ---------
        objects (list): List of panel components to add as tabs.
        """
        new_objects, new_names = self._to_objects_and_names(panes)
        objects = list(self)
        objects.extend(new_objects)
        self._names.extend(new_names)
        self.objects = objects

    def insert(self, index: int, pane: Any) -> None:
        """
        Inserts an object in the tabs at the specified index.

        Arguments
        ---------
        index (int): Index at which to insert the object.
        object (object): Panel components to insert as tabs.
        """
        new_object, new_name = self._to_object_and_name(pane)
        new_objects = list(self.objects)
        new_objects.insert(index, new_object)
        self._names.insert(index, new_name)
        self.objects = new_objects

    def pop(self, index: int) -> Viewable:
        """
        Pops an item from the tabs by index.

        Arguments
        ---------
        index (int): The index of the item to pop from the tabs.
        """
        new_objects = list(self)
        obj = new_objects.pop(index)
        self._names.pop(index)
        self.objects = new_objects
        return obj

    def remove(self, pane: Viewable) -> None:
        """
        Removes an object from the tabs.

        Arguments
        ---------
        obj (object): The object to remove from the tabs.
        """
        new_objects = list(self)
        if pane in new_objects:
            index = new_objects.index(pane)
        new_objects.remove(pane)
        self._names.pop(index)
        self.objects = new_objects

    def reverse(self) -> None:
        """
        Reverses the tabs.
        """
        new_objects = list(self)
        new_objects.reverse()
        self._names.reverse()
        self.objects = new_objects