import itertools
import weakref
from zope.interface import Interface
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface import ro
from zope.interface._compat import _normalize_name
from zope.interface._compat import _use_c_impl
from zope.interface.interfaces import IAdapterRegistry
class BaseAdapterRegistry:
    """
    A basic implementation of the data storage and algorithms required
    for a :class:`zope.interface.interfaces.IAdapterRegistry`.

    Subclasses can set the following attributes to control how the data
    is stored; in particular, these hooks can be helpful for ZODB
    persistence. They can be class attributes that are the named (or similar) type, or
    they can be methods that act as a constructor for an object that behaves
    like the types defined here; this object will not assume that they are type
    objects, but subclasses are free to do so:

    _sequenceType = list
      This is the type used for our two mutable top-level "byorder" sequences.
      Must support mutation operations like ``append()`` and ``del seq[index]``.
      These are usually small (< 10). Although at least one of them is
      accessed when performing lookups or queries on this object, the other
      is untouched. In many common scenarios, both are only required when
      mutating registrations and subscriptions (like what
      :meth:`zope.interface.interfaces.IComponents.registerUtility` does).
      This use pattern makes it an ideal candidate to be a
      :class:`~persistent.list.PersistentList`.
    _leafSequenceType = tuple
      This is the type used for the leaf sequences of subscribers.
      It could be set to a ``PersistentList`` to avoid many unnecessary data
      loads when subscribers aren't being used. Mutation operations are directed
      through :meth:`_addValueToLeaf` and :meth:`_removeValueFromLeaf`; if you use
      a mutable type, you'll need to override those.
    _mappingType = dict
      This is the mutable mapping type used for the keyed mappings.
      A :class:`~persistent.mapping.PersistentMapping`
      could be used to help reduce the number of data loads when the registry is large
      and parts of it are rarely used. Further reductions in data loads can come from
      using a :class:`~BTrees.OOBTree.OOBTree`, but care is required
      to be sure that all required/provided
      values are fully ordered (e.g., no required or provided values that are classes
      can be used).
    _providedType = dict
      This is the mutable mapping type used for the ``_provided`` mapping.
      This is separate from the generic mapping type because the values
      are always integers, so one might choose to use a more optimized data
      structure such as a :class:`~BTrees.OIBTree.OIBTree`.
      The same caveats regarding key types
      apply as for ``_mappingType``.

    It is possible to also set these on an instance, but because of the need to
    potentially also override :meth:`_addValueToLeaf` and :meth:`_removeValueFromLeaf`,
    this may be less useful in a persistent scenario; using a subclass is recommended.

    .. versionchanged:: 5.3.0
        Add support for customizing the way internal data
        structures are created.
    .. versionchanged:: 5.3.0
        Add methods :meth:`rebuild`, :meth:`allRegistrations`
        and :meth:`allSubscriptions`.
    """
    _delegated = ('lookup', 'queryMultiAdapter', 'lookup1', 'queryAdapter', 'adapter_hook', 'lookupAll', 'names', 'subscriptions', 'subscribers')
    _generation = 0

    def __init__(self, bases=()):
        self._adapters = self._sequenceType()
        self._subscribers = self._sequenceType()
        self._provided = self._providedType()
        self._createLookup()
        self.__bases__ = bases

    def _setBases(self, bases):
        """
        If subclasses need to track when ``__bases__`` changes, they
        can override this method.

        Subclasses must still call this method.
        """
        self.__dict__['__bases__'] = bases
        self.ro = ro.ro(self)
        self.changed(self)
    __bases__ = property(lambda self: self.__dict__['__bases__'], lambda self, bases: self._setBases(bases))

    def _createLookup(self):
        self._v_lookup = self.LookupClass(self)
        for name in self._delegated:
            self.__dict__[name] = getattr(self._v_lookup, name)
    _sequenceType = list
    _leafSequenceType = tuple
    _mappingType = dict
    _providedType = dict

    def _addValueToLeaf(self, existing_leaf_sequence, new_item):
        """
        Add the value *new_item* to the *existing_leaf_sequence*, which may
        be ``None``.

        Subclasses that redefine `_leafSequenceType` should override this method.

        :param existing_leaf_sequence:
            If *existing_leaf_sequence* is not *None*, it will be an instance
            of `_leafSequenceType`. (Unless the object has been unpickled
            from an old pickle and the class definition has changed, in which case
            it may be an instance of a previous definition, commonly a `tuple`.)

        :return:
           This method returns the new value to be stored. It may mutate the
           sequence in place if it was not ``None`` and the type is mutable, but
           it must also return it.

        .. versionadded:: 5.3.0
        """
        if existing_leaf_sequence is None:
            return (new_item,)
        return existing_leaf_sequence + (new_item,)

    def _removeValueFromLeaf(self, existing_leaf_sequence, to_remove):
        """
        Remove the item *to_remove* from the (non-``None``, non-empty)
        *existing_leaf_sequence* and return the mutated sequence.

        If there is more than one item that is equal to *to_remove*
        they must all be removed.

        Subclasses that redefine `_leafSequenceType` should override
        this method. Note that they can call this method to help
        in their implementation; this implementation will always
        return a new tuple constructed by iterating across
        the *existing_leaf_sequence* and omitting items equal to *to_remove*.

        :param existing_leaf_sequence:
           As for `_addValueToLeaf`, probably an instance of
           `_leafSequenceType` but possibly an older type; never `None`.
        :return:
           A version of *existing_leaf_sequence* with all items equal to
           *to_remove* removed. Must not return `None`. However,
           returning an empty
           object, even of another type such as the empty tuple, ``()`` is
           explicitly allowed; such an object will never be stored.

        .. versionadded:: 5.3.0
        """
        return tuple([v for v in existing_leaf_sequence if v != to_remove])

    def changed(self, originally_changed):
        self._generation += 1
        self._v_lookup.changed(originally_changed)

    def register(self, required, provided, name, value):
        if not isinstance(name, str):
            raise ValueError('name is not a string')
        if value is None:
            self.unregister(required, provided, name, value)
            return
        required = tuple([_convert_None_to_Interface(r) for r in required])
        name = _normalize_name(name)
        order = len(required)
        byorder = self._adapters
        while len(byorder) <= order:
            byorder.append(self._mappingType())
        components = byorder[order]
        key = required + (provided,)
        for k in key:
            d = components.get(k)
            if d is None:
                d = self._mappingType()
                components[k] = d
            components = d
        if components.get(name) is value:
            return
        components[name] = value
        n = self._provided.get(provided, 0) + 1
        self._provided[provided] = n
        if n == 1:
            self._v_lookup.add_extendor(provided)
        self.changed(self)

    def _find_leaf(self, byorder, required, provided, name):
        required = tuple([_convert_None_to_Interface(r) for r in required])
        order = len(required)
        if len(byorder) <= order:
            return None
        components = byorder[order]
        key = required + (provided,)
        for k in key:
            d = components.get(k)
            if d is None:
                return None
            components = d
        return components.get(name)

    def registered(self, required, provided, name=''):
        return self._find_leaf(self._adapters, required, provided, _normalize_name(name))

    @classmethod
    def _allKeys(cls, components, i, parent_k=()):
        if i == 0:
            for k, v in components.items():
                yield (parent_k + (k,), v)
        else:
            for k, v in components.items():
                new_parent_k = parent_k + (k,)
                yield from cls._allKeys(v, i - 1, new_parent_k)

    def _all_entries(self, byorder):
        for i, components in enumerate(byorder):
            for key, value in self._allKeys(components, i + 1):
                assert len(key) == i + 2
                required = key[:i]
                provided = key[-2]
                name = key[-1]
                yield (required, provided, name, value)

    def allRegistrations(self):
        """
        Yields tuples ``(required, provided, name, value)`` for all
        the registrations that this object holds.

        These tuples could be passed as the arguments to the
        :meth:`register` method on another adapter registry to
        duplicate the registrations this object holds.

        .. versionadded:: 5.3.0
        """
        yield from self._all_entries(self._adapters)

    def unregister(self, required, provided, name, value=None):
        required = tuple([_convert_None_to_Interface(r) for r in required])
        order = len(required)
        byorder = self._adapters
        if order >= len(byorder):
            return False
        components = byorder[order]
        key = required + (provided,)
        lookups = []
        for k in key:
            d = components.get(k)
            if d is None:
                return
            lookups.append((components, k))
            components = d
        old = components.get(name)
        if old is None:
            return
        if value is not None and old is not value:
            return
        del components[name]
        if not components:
            for comp, k in reversed(lookups):
                d = comp[k]
                if d:
                    break
                else:
                    del comp[k]
            while byorder and (not byorder[-1]):
                del byorder[-1]
        n = self._provided[provided] - 1
        if n == 0:
            del self._provided[provided]
            self._v_lookup.remove_extendor(provided)
        else:
            self._provided[provided] = n
        self.changed(self)

    def subscribe(self, required, provided, value):
        required = tuple([_convert_None_to_Interface(r) for r in required])
        name = ''
        order = len(required)
        byorder = self._subscribers
        while len(byorder) <= order:
            byorder.append(self._mappingType())
        components = byorder[order]
        key = required + (provided,)
        for k in key:
            d = components.get(k)
            if d is None:
                d = self._mappingType()
                components[k] = d
            components = d
        components[name] = self._addValueToLeaf(components.get(name), value)
        if provided is not None:
            n = self._provided.get(provided, 0) + 1
            self._provided[provided] = n
            if n == 1:
                self._v_lookup.add_extendor(provided)
        self.changed(self)

    def subscribed(self, required, provided, subscriber):
        subscribers = self._find_leaf(self._subscribers, required, provided, '') or ()
        return subscriber if subscriber in subscribers else None

    def allSubscriptions(self):
        """
        Yields tuples ``(required, provided, value)`` for all the
        subscribers that this object holds.

        These tuples could be passed as the arguments to the
        :meth:`subscribe` method on another adapter registry to
        duplicate the registrations this object holds.

        .. versionadded:: 5.3.0
        """
        for required, provided, _name, value in self._all_entries(self._subscribers):
            for v in value:
                yield (required, provided, v)

    def unsubscribe(self, required, provided, value=None):
        required = tuple([_convert_None_to_Interface(r) for r in required])
        order = len(required)
        byorder = self._subscribers
        if order >= len(byorder):
            return
        components = byorder[order]
        key = required + (provided,)
        lookups = []
        for k in key:
            d = components.get(k)
            if d is None:
                return
            lookups.append((components, k))
            components = d
        old = components.get('')
        if not old:
            return
        len_old = len(old)
        if value is None:
            new = ()
        else:
            new = self._removeValueFromLeaf(old, value)
        del old
        if len(new) == len_old:
            return
        if new:
            components[''] = new
        else:
            del components['']
            for comp, k in reversed(lookups):
                d = comp[k]
                if d:
                    break
                else:
                    del comp[k]
            while byorder and (not byorder[-1]):
                del byorder[-1]
        if provided is not None:
            n = self._provided[provided] + len(new) - len_old
            if n == 0:
                del self._provided[provided]
                self._v_lookup.remove_extendor(provided)
            else:
                self._provided[provided] = n
        self.changed(self)

    def rebuild(self):
        """
        Rebuild (and replace) all the internal data structures of this
        object.

        This is useful, especially for persistent implementations, if
        you suspect an issue with reference counts keeping interfaces
        alive even though they are no longer used.

        It is also useful if you or a subclass change the data types
        (``_mappingType`` and friends) that are to be used.

        This method replaces all internal data structures with new objects;
        it specifically does not re-use any storage.

        .. versionadded:: 5.3.0
        """
        registrations = self.allRegistrations()
        subscriptions = self.allSubscriptions()

        def buffer(it):
            try:
                first = next(it)
            except StopIteration:
                return iter(())
            return itertools.chain((first,), it)
        registrations = buffer(registrations)
        subscriptions = buffer(subscriptions)
        self.__init__(self.__bases__)
        for args in registrations:
            self.register(*args)
        for args in subscriptions:
            self.subscribe(*args)

    def get(self, _):

        class XXXTwistedFakeOut:
            selfImplied = {}
        return XXXTwistedFakeOut