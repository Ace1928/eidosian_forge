import collections.abc
import copy
import pickle
from typing import (
class RepeatedCompositeFieldContainer(BaseContainer[_T], MutableSequence[_T]):
    """Simple, list-like container for holding repeated composite fields."""
    __slots__ = ['_message_descriptor']

    def __init__(self, message_listener: Any, message_descriptor: Any) -> None:
        """
    Note that we pass in a descriptor instead of the generated directly,
    since at the time we construct a _RepeatedCompositeFieldContainer we
    haven't yet necessarily initialized the type that will be contained in the
    container.

    Args:
      message_listener: A MessageListener implementation.
        The RepeatedCompositeFieldContainer will call this object's
        Modified() method when it is modified.
      message_descriptor: A Descriptor instance describing the protocol type
        that should be present in this container.  We'll use the
        _concrete_class field of this descriptor when the client calls add().
    """
        super().__init__(message_listener)
        self._message_descriptor = message_descriptor

    def add(self, **kwargs: Any) -> _T:
        """Adds a new element at the end of the list and returns it. Keyword
    arguments may be used to initialize the element.
    """
        new_element = self._message_descriptor._concrete_class(**kwargs)
        new_element._SetListener(self._message_listener)
        self._values.append(new_element)
        if not self._message_listener.dirty:
            self._message_listener.Modified()
        return new_element

    def append(self, value: _T) -> None:
        """Appends one element by copying the message."""
        new_element = self._message_descriptor._concrete_class()
        new_element._SetListener(self._message_listener)
        new_element.CopyFrom(value)
        self._values.append(new_element)
        if not self._message_listener.dirty:
            self._message_listener.Modified()

    def insert(self, key: int, value: _T) -> None:
        """Inserts the item at the specified position by copying."""
        new_element = self._message_descriptor._concrete_class()
        new_element._SetListener(self._message_listener)
        new_element.CopyFrom(value)
        self._values.insert(key, new_element)
        if not self._message_listener.dirty:
            self._message_listener.Modified()

    def extend(self, elem_seq: Iterable[_T]) -> None:
        """Extends by appending the given sequence of elements of the same type

    as this one, copying each individual message.
    """
        message_class = self._message_descriptor._concrete_class
        listener = self._message_listener
        values = self._values
        for message in elem_seq:
            new_element = message_class()
            new_element._SetListener(listener)
            new_element.MergeFrom(message)
            values.append(new_element)
        listener.Modified()

    def MergeFrom(self, other: Union['RepeatedCompositeFieldContainer[_T]', Iterable[_T]]) -> None:
        """Appends the contents of another repeated field of the same type to this
    one, copying each individual message.
    """
        self.extend(other)

    def remove(self, elem: _T) -> None:
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

    def __setitem__(self, key, value):
        raise TypeError(f'{self.__class__.__name__} object does not support item assignment')

    def __delitem__(self, key: Union[int, slice]) -> None:
        """Deletes the item at the specified position."""
        del self._values[key]
        self._message_listener.Modified()

    def __eq__(self, other: Any) -> bool:
        """Compares the current instance with another one."""
        if self is other:
            return True
        if not isinstance(other, self.__class__):
            raise TypeError('Can only compare repeated composite fields against other repeated composite fields.')
        return self._values == other._values