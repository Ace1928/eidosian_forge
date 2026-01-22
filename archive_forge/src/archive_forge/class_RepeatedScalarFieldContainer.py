import collections.abc
import copy
import pickle
from typing import (
class RepeatedScalarFieldContainer(BaseContainer[_T], MutableSequence[_T]):
    """Simple, type-checked, list-like container for holding repeated scalars."""
    __slots__ = ['_type_checker']

    def __init__(self, message_listener: Any, type_checker: Any) -> None:
        """Args:

      message_listener: A MessageListener implementation. The
      RepeatedScalarFieldContainer will call this object's Modified() method
      when it is modified.
      type_checker: A type_checkers.ValueChecker instance to run on elements
      inserted into this container.
    """
        super().__init__(message_listener)
        self._type_checker = type_checker

    def append(self, value: _T) -> None:
        """Appends an item to the list. Similar to list.append()."""
        self._values.append(self._type_checker.CheckValue(value))
        if not self._message_listener.dirty:
            self._message_listener.Modified()

    def insert(self, key: int, value: _T) -> None:
        """Inserts the item at the specified position. Similar to list.insert()."""
        self._values.insert(key, self._type_checker.CheckValue(value))
        if not self._message_listener.dirty:
            self._message_listener.Modified()

    def extend(self, elem_seq: Iterable[_T]) -> None:
        """Extends by appending the given iterable. Similar to list.extend()."""
        elem_seq_iter = iter(elem_seq)
        new_values = [self._type_checker.CheckValue(elem) for elem in elem_seq_iter]
        if new_values:
            self._values.extend(new_values)
        self._message_listener.Modified()

    def MergeFrom(self, other: Union['RepeatedScalarFieldContainer[_T]', Iterable[_T]]) -> None:
        """Appends the contents of another repeated field of the same type to this
    one. We do not check the types of the individual fields.
    """
        self._values.extend(other)
        self._message_listener.Modified()

    def remove(self, elem: _T):
        """Removes an item from the list. Similar to list.remove()."""
        self._values.remove(elem)
        self._message_listener.Modified()

    def pop(self, key: Optional[int]=-1) -> _T:
        """Removes and returns an item at a given index. Similar to list.pop()."""
        value = self._values[key]
        self.__delitem__(key)
        return value

    @overload
    def __setitem__(self, key: int, value: _T) -> None:
        ...

    @overload
    def __setitem__(self, key: slice, value: Iterable[_T]) -> None:
        ...

    def __setitem__(self, key, value) -> None:
        """Sets the item on the specified position."""
        if isinstance(key, slice):
            if key.step is not None:
                raise ValueError('Extended slices not supported')
            self._values[key] = map(self._type_checker.CheckValue, value)
            self._message_listener.Modified()
        else:
            self._values[key] = self._type_checker.CheckValue(value)
            self._message_listener.Modified()

    def __delitem__(self, key: Union[int, slice]) -> None:
        """Deletes the item at the specified position."""
        del self._values[key]
        self._message_listener.Modified()

    def __eq__(self, other: Any) -> bool:
        """Compares the current instance with another one."""
        if self is other:
            return True
        if isinstance(other, self.__class__):
            return other._values == self._values
        return other == self._values

    def __deepcopy__(self, unused_memo: Any=None) -> 'RepeatedScalarFieldContainer[_T]':
        clone = RepeatedScalarFieldContainer(copy.deepcopy(self._message_listener), self._type_checker)
        clone.MergeFrom(self)
        return clone

    def __reduce__(self, **kwargs) -> NoReturn:
        raise pickle.PickleError("Can't pickle repeated scalar fields, convert to list first")