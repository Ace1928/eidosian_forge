import sys
import weakref
from types import FunctionType
from types import MethodType
from typing import Union
from zope.interface import ro
from zope.interface._compat import _use_c_impl
from zope.interface.exceptions import Invalid
from zope.interface.ro import ro as calculate_ro
from zope.interface.declarations import implementedBy
from zope.interface.declarations import providedBy
from zope.interface.exceptions import BrokenImplementation
from zope.interface.exceptions import InvalidInterface
from zope.interface.declarations import _empty
class InterfaceClass(_InterfaceClassBase):
    """
    Prototype (scarecrow) Interfaces Implementation.

    Note that it is not possible to change the ``__name__`` or ``__module__``
    after an instance of this object has been constructed.
    """

    def __new__(cls, name=None, bases=(), attrs=None, __doc__=None, __module__=None):
        assert isinstance(bases, tuple)
        attrs = attrs or {}
        needs_custom_class = attrs.pop(INTERFACE_METHODS, None)
        if needs_custom_class:
            needs_custom_class.update({'__classcell__': attrs.pop('__classcell__')} if '__classcell__' in attrs else {})
            if '__adapt__' in needs_custom_class:
                needs_custom_class['_CALL_CUSTOM_ADAPT'] = 1
            if issubclass(cls, _InterfaceClassWithCustomMethods):
                cls_bases = (cls,)
            elif cls is InterfaceClass:
                cls_bases = (_InterfaceClassWithCustomMethods,)
            else:
                cls_bases = (cls, _InterfaceClassWithCustomMethods)
            cls = type(cls)(name + '<WithCustomMethods>', cls_bases, needs_custom_class)
        return _InterfaceClassBase.__new__(cls)

    def __init__(self, name, bases=(), attrs=None, __doc__=None, __module__=None):
        if not all((isinstance(base, InterfaceClass) for base in bases)):
            raise TypeError('Expected base interfaces')
        if attrs is None:
            attrs = {}
        if __module__ is None:
            __module__ = attrs.get('__module__')
            if isinstance(__module__, str):
                del attrs['__module__']
            else:
                try:
                    __module__ = sys._getframe(1).f_globals['__name__']
                except (AttributeError, KeyError):
                    pass
        InterfaceBase.__init__(self, name, __module__)
        d = attrs.get('__doc__')
        if d is not None:
            if not isinstance(d, Attribute):
                if __doc__ is None:
                    __doc__ = d
                del attrs['__doc__']
        if __doc__ is None:
            __doc__ = ''
        Element.__init__(self, name, __doc__)
        tagged_data = attrs.pop(TAGGED_DATA, None)
        if tagged_data is not None:
            for key, val in tagged_data.items():
                self.setTaggedValue(key, val)
        Specification.__init__(self, bases)
        self.__attrs = self.__compute_attrs(attrs)
        self.__identifier__ = '{}.{}'.format(__module__, name)

    def __compute_attrs(self, attrs):

        def update_value(aname, aval):
            if isinstance(aval, Attribute):
                aval.interface = self
                if not aval.__name__:
                    aval.__name__ = aname
            elif isinstance(aval, FunctionType):
                aval = fromFunction(aval, self, name=aname)
            else:
                raise InvalidInterface('Concrete attribute, ' + aname)
            return aval
        return {aname: update_value(aname, aval) for aname, aval in attrs.items() if aname not in ('__locals__', '__qualname__', '__annotations__') and aval is not _decorator_non_return}

    def interfaces(self):
        """Return an iterator for the interfaces in the specification.
        """
        yield self

    def getBases(self):
        return self.__bases__

    def isEqualOrExtendedBy(self, other):
        """Same interface or extends?"""
        return self == other or other.extends(self)

    def names(self, all=False):
        """Return the attribute names defined by the interface."""
        if not all:
            return self.__attrs.keys()
        r = self.__attrs.copy()
        for base in self.__bases__:
            r.update(dict.fromkeys(base.names(all)))
        return r.keys()

    def __iter__(self):
        return iter(self.names(all=True))

    def namesAndDescriptions(self, all=False):
        """Return attribute names and descriptions defined by interface."""
        if not all:
            return self.__attrs.items()
        r = {}
        for base in self.__bases__[::-1]:
            r.update(dict(base.namesAndDescriptions(all)))
        r.update(self.__attrs)
        return r.items()

    def getDescriptionFor(self, name):
        """Return the attribute description for the given name."""
        r = self.get(name)
        if r is not None:
            return r
        raise KeyError(name)
    __getitem__ = getDescriptionFor

    def __contains__(self, name):
        return self.get(name) is not None

    def direct(self, name):
        return self.__attrs.get(name)

    def queryDescriptionFor(self, name, default=None):
        return self.get(name, default)

    def validateInvariants(self, obj, errors=None):
        """validate object to defined invariants."""
        for iface in self.__iro__:
            for invariant in iface.queryDirectTaggedValue('invariants', ()):
                try:
                    invariant(obj)
                except Invalid as error:
                    if errors is not None:
                        errors.append(error)
                    else:
                        raise
        if errors:
            raise Invalid(errors)

    def queryTaggedValue(self, tag, default=None):
        """
        Queries for the value associated with *tag*, returning it from the nearest
        interface in the ``__iro__``.

        If not found, returns *default*.
        """
        for iface in self.__iro__:
            value = iface.queryDirectTaggedValue(tag, _marker)
            if value is not _marker:
                return value
        return default

    def getTaggedValue(self, tag):
        """ Returns the value associated with 'tag'. """
        value = self.queryTaggedValue(tag, default=_marker)
        if value is _marker:
            raise KeyError(tag)
        return value

    def getTaggedValueTags(self):
        """ Returns a list of all tags. """
        keys = set()
        for base in self.__iro__:
            keys.update(base.getDirectTaggedValueTags())
        return keys

    def __repr__(self):
        try:
            return self._v_repr
        except AttributeError:
            name = str(self)
            r = '<{} {}>'.format(self.__class__.__name__, name)
            self._v_repr = r
            return r

    def __str__(self):
        name = self.__name__
        m = self.__ibmodule__
        if m:
            name = '{}.{}'.format(m, name)
        return name

    def _call_conform(self, conform):
        try:
            return conform(self)
        except TypeError:
            if sys.exc_info()[2].tb_next is not None:
                raise
        return None

    def __reduce__(self):
        return self.__name__

    def __or__(self, other):
        """Allow type hinting syntax: Interface | None."""
        return Union[self, other]

    def __ror__(self, other):
        """Allow type hinting syntax: None | Interface."""
        return Union[other, self]