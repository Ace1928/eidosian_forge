import copy
import operator
from weakref import ref
from traits.observation.i_observable import IObservable
from traits.trait_base import class_of, Undefined, _validate_everything
from traits.trait_errors import TraitError
@IObservable.register
class TraitList(list):
    """ A subclass of list that validates and notifies listeners of changes.

    Parameters
    ----------
    value : iterable
        Iterable providing the items for the list
    item_validator : callable, optional
        Called to validate and/or transform items added to the list. The
        callable should accept a single item from the list and return
        the transformed item, raising TraitError for invalid items. If
        not given, no item validation is performed.
    notifiers : list of callable, optional
        A list of callables with the signature::

            notifier(trait_list, index, removed, added)

        If this argument is not given, the list of notifiers is initially
        empty.

    Attributes
    ----------
    item_validator : callable
        Called to validate and/or transform items added to the list. The
        callable should accept a single item from the list and return
        the transformed item, raising TraitError for invalid items.
    notifiers : list of callable
        A list of callables with the signature::

            notifier(trait_list, index, removed, added)
    """

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        self.item_validator = _validate_everything
        self.notifiers = []
        return self

    def __init__(self, iterable=(), *, item_validator=None, notifiers=None):
        if item_validator is not None:
            self.item_validator = item_validator
        super().__init__((self.item_validator(item) for item in iterable))
        if notifiers is not None:
            self.notifiers = list(notifiers)

    def notify(self, index, removed, added):
        """ Call all notifiers.

        This simply calls all notifiers provided by the class, if any.
        The notifiers are expected to have the signature::

            notifier(trait_list, index, removed, added)

        Any return values are ignored.

        Parameters
        ----------
        index : int or slice
            The indices being modified by the operation.
        removed : list
            The items to be removed.
        added : list
            The items being added to the list.
        """
        for notifier in self.notifiers:
            notifier(self, index, removed, added)

    def __delitem__(self, key):
        """ Delete self[key].

        Parameters
        ----------
        key : integer or slice
            Index of the element(s) to be deleted.

        Raises
        ------
        IndexError
            If key is an integer index and is out of range.
        """
        original_length = len(self)
        removed = _removed_items(self, key, return_for_invalid_index=None)
        super().__delitem__(key)
        if removed:
            reversed, normalized_key = _normalize_slice_or_index(key, original_length)
            if reversed:
                removed = removed[::-1]
            self.notify(normalized_key, removed, [])

    def __iadd__(self, value):
        """ Implement self += value.

        Parameters
        ----------
        value : iterable
            The items to be added.

        Returns
        -------
        self : TraitList
            The modified list.
        """
        original_length = len(self)
        added = [self.item_validator(item) for item in value]
        extended = super().__iadd__(added)
        if added:
            self.notify(original_length, [], added)
        return extended

    def __imul__(self, value):
        """ Implement self *= value.

        Parameters
        ----------
        value : integer
            The multiplier.

        Returns
        -------
        self : TraitList
            The modified list.
        """
        if value < 1:
            removed = self.copy()
            multiplied = super().__imul__(value)
            if removed:
                self.notify(0, removed, [])
        else:
            original_length = len(self)
            multiplied = super().__imul__(value)
            added = self[original_length:]
            if added:
                self.notify(original_length, [], added)
        return multiplied

    def __setitem__(self, key, value):
        """ Set self[key] to value.

        Parameters
        ----------
        key : integer or slice
            Index of the element(s) to be replaced.
        value : iterable
            Replacement values.

        Raises
        ------
        IndexError
            If key is an integer index and is out of range.
        ValueError
            If key is an extended slice (that is, it's a slice whose step
            is not 1 and not None) and the number of replacement elements
            doesn't match the number of removed elements.
        """
        original_length = len(self)
        removed = _removed_items(self, key, return_for_invalid_index=None)
        if isinstance(key, slice):
            value = [self.item_validator(item) for item in value]
            added = value
        else:
            value = self.item_validator(value)
            added = [value]
        super().__setitem__(key, value)
        if added or removed:
            reversed, normalized_key = _normalize_slice_or_index(key, original_length)
            if reversed:
                added = added[::-1]
                removed = removed[::-1]
            self.notify(normalized_key, removed, added)

    def append(self, object):
        """ Append object to the end of the list.

        Parameters
        ----------
        object : any
            The object to append.
        """
        original_length = len(self)
        super().append(self.item_validator(object))
        self.notify(original_length, [], self[original_length:])

    def clear(self):
        """ Remove all items from list. """
        removed = self.copy()
        super().clear()
        if removed:
            self.notify(0, removed, [])

    def extend(self, iterable):
        """ Extend list by appending elements from the iterable.

        Parameters
        ----------
        iterable : iterable
            The elements to append.
        """
        original_length = len(self)
        added = [self.item_validator(item) for item in iterable]
        super().extend(added)
        if added:
            self.notify(original_length, [], added)

    def insert(self, index, object):
        """ Insert object before index.

        Parameters
        ----------
        index : integer
            The position at which to insert.
        object : object
            The object to insert.
        """
        if index < 0:
            normalized_index = max(index + len(self), 0)
        else:
            normalized_index = min(index, len(self))
        object = self.item_validator(object)
        super().insert(index, self.item_validator(object))
        self.notify(normalized_index, [], [object])

    def pop(self, index=-1):
        """ Remove and return item at index (default last).

        Parameters
        ----------
        index : int, optional
            Index at which to remove item. If not given, the
            last item of the list is removed.

        Returns
        -------
        item : object
            The removed item.

        Raises
        ------
        IndexError
            If list is empty or index is out of range.
        """
        normalized_index = index + len(self) if index < 0 else index
        item = super().pop(index)
        self.notify(normalized_index, [item], [])
        return item

    def remove(self, value):
        """ Remove first occurrence of value.

        Notes
        -----
        The value is not validated or converted before removal.

        Parameters
        ----------
        value : object
            Value to be removed.

        Raises
        ------
        ValueError
            If the value is not present.
        """
        try:
            index = self.index(value)
        except ValueError:
            pass
        else:
            removed = [self[index]]
        super().remove(value)
        self.notify(index, removed, [])

    def reverse(self):
        """ Reverse the items in the list in place. """
        removed = self.copy()
        super().reverse()
        if removed:
            self.notify(0, removed, self.copy())

    def sort(self, *, key=None, reverse=False):
        """ Sort the list in ascending order and return None.

        The sort is in-place (i.e. the list itself is modified) and stable
        (i.e. the order of two equal elements is maintained).

        If a key function is given, apply it once to each list item and sort
        them, ascending or descending, according to their function values.

        The reverse flag can be set to sort in descending order.

        Parameters
        ----------
        key : callable
            Custom function that accepts a single item from the list and
            returns the key to be used in comparisons.
        reverse : bool
            If true, the resulting list will be sorted in descending order.
        """
        removed = self.copy()
        super().sort(key=key, reverse=reverse)
        if removed:
            self.notify(0, removed, self.copy())

    def __deepcopy__(self, memo):
        """ Perform a deepcopy operation.

        Notifiers are transient and should not be copied.
        """
        return type(self)([copy.deepcopy(x, memo) for x in self], item_validator=copy.deepcopy(self.item_validator, memo))

    def __getstate__(self):
        """ Get the state of the object for serialization.

        Notifiers are transient and should not be serialized.
        """
        result = self.__dict__.copy()
        result.pop('notifiers', None)
        return result

    def __setstate__(self, state):
        """ Restore the state of the object after serialization.

        Notifiers are transient and are restored to the empty list.
        """
        state['notifiers'] = []
        self.__dict__.update(state)

    def _notifiers(self, force_create):
        """ Return a list of callables where each callable is a notifier.
        The list is expected to be mutated for contributing or removing
        notifiers from the object.

        Parameters
        ----------
        force_create: boolean
            It is added for compatibility with CTrait. Not used here.
        """
        return self.notifiers