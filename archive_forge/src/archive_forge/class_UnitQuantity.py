import weakref
import numpy
from .dimensionality import Dimensionality
from . import markup
from .quantity import Quantity, get_conversion_factor
from .registry import unit_registry
from .decorators import memoize, with_doc
class UnitQuantity(Quantity):
    _primary_order = 90
    _secondary_order = 0
    _reference_quantity = None
    __array_priority__ = 20

    def __new__(cls, name, definition=None, symbol=None, u_symbol=None, aliases=[], doc=None):
        try:
            assert isinstance(name, str)
        except AssertionError:
            raise TypeError('name must be a string, got %s (not unicode)' % name)
        try:
            assert symbol is None or isinstance(symbol, str)
        except AssertionError:
            raise TypeError('symbol must be a string, got %s (u_symbol can be unicode)' % symbol)
        ret = numpy.array(1, dtype='d').view(cls)
        ret.flags.writeable = False
        ret._name = name
        ret._symbol = symbol
        ret._u_symbol = u_symbol
        if doc is not None:
            ret.__doc__ = doc
        if definition is not None:
            if not isinstance(definition, Quantity):
                definition *= dimensionless
            ret._definition = definition
            ret._conv_ref = definition._reference
        else:
            ret._definition = None
            ret._conv_ref = None
        ret._aliases = aliases
        ret._format_order = (ret._primary_order, ret._secondary_order)
        ret.__class__._secondary_order += 1
        return ret

    def __init__(self, name, definition=None, symbol=None, u_symbol=None, aliases=[], doc=None):
        unit_registry[name] = self
        if symbol:
            unit_registry[symbol] = self
        for alias in aliases:
            unit_registry[alias] = self

    def __array_finalize__(self, obj):
        pass

    def __hash__(self):
        return hash((type(self), self._name))

    @property
    def _reference(self):
        if self._conv_ref is None:
            return self
        else:
            return self._conv_ref

    @property
    def _dimensionality(self):
        return Dimensionality({self: 1})

    @property
    def format_order(self):
        return self._format_order

    @property
    def name(self):
        return self._name

    @property
    def definition(self):
        if self._definition is None:
            return self
        else:
            return self._definition

    @property
    def simplified(self):
        return self._reference.simplified

    @property
    def symbol(self):
        if self._symbol:
            return self._symbol
        else:
            return self.name

    @property
    def u_symbol(self):
        if self._u_symbol:
            return self._u_symbol
        else:
            return self.symbol

    @property
    def units(self):
        return self

    @units.setter
    def units(self, units):
        raise AttributeError('can not modify protected units')

    def __repr__(self):
        ref = self._definition
        if ref:
            ref = ', %s * %s' % (str(ref.magnitude), ref.dimensionality.string)
        else:
            ref = ''
        symbol = self._symbol
        symbol = ', %s' % repr(symbol) if symbol else ''
        if markup.config.use_unicode:
            u_symbol = self._u_symbol
            u_symbol = ', %s' % repr(u_symbol) if u_symbol else ''
        else:
            u_symbol = ''
        return '%s(%s%s%s%s)' % (self.__class__.__name__, repr(self.name), ref, symbol, u_symbol)

    @with_doc(Quantity.__str__, use_header=False)
    def __str__(self):
        if self.u_symbol != self.name:
            if markup.config.use_unicode:
                s = '1 %s (%s)' % (self.u_symbol, self.name)
            else:
                s = '1 %s (%s)' % (self.symbol, self.name)
        else:
            s = '1 %s' % self.name
        return s

    @with_doc(Quantity.__add__, use_header=False)
    def __add__(self, other):
        return self.view(Quantity).__add__(other)

    @with_doc(Quantity.__radd__, use_header=False)
    def __radd__(self, other):
        try:
            return self.rescale(other.units).__radd__(other)
        except AttributeError:
            return self.view(Quantity).__radd__(other)

    @with_doc(Quantity.__sub__, use_header=False)
    def __sub__(self, other):
        return self.view(Quantity).__sub__(other)

    @with_doc(Quantity.__rsub__, use_header=False)
    def __rsub__(self, other):
        try:
            return self.rescale(other.units).__rsub__(other)
        except AttributeError:
            return self.view(Quantity).__rsub__(other)

    @with_doc(Quantity.__mod__, use_header=False)
    def __mod__(self, other):
        return self.view(Quantity).__mod__(other)

    @with_doc(Quantity.__rsub__, use_header=False)
    def __rmod__(self, other):
        try:
            return self.rescale(other.units).__rmod__(other)
        except AttributeError:
            return self.view(Quantity).__rmod__(other)

    @with_doc(Quantity.__mul__, use_header=False)
    def __mul__(self, other):
        return self.view(Quantity).__mul__(other)

    @with_doc(Quantity.__rmul__, use_header=False)
    def __rmul__(self, other):
        return self.view(Quantity).__rmul__(other)

    @with_doc(Quantity.__truediv__, use_header=False)
    def __truediv__(self, other):
        return self.view(Quantity).__truediv__(other)

    @with_doc(Quantity.__rtruediv__, use_header=False)
    def __rtruediv__(self, other):
        return self.view(Quantity).__rtruediv__(other)

    @with_doc(Quantity.__pow__, use_header=False)
    def __pow__(self, other):
        return self.view(Quantity).__pow__(other)

    @with_doc(Quantity.__rpow__, use_header=False)
    def __rpow__(self, other):
        return self.view(Quantity).__rpow__(other)

    @with_doc(Quantity.__iadd__, use_header=False)
    def __iadd__(self, other):
        raise TypeError('can not modify protected units')

    @with_doc(Quantity.__isub__, use_header=False)
    def __isub__(self, other):
        raise TypeError('can not modify protected units')

    @with_doc(Quantity.__imul__, use_header=False)
    def __imul__(self, other):
        raise TypeError('can not modify protected units')

    @with_doc(Quantity.__itruediv__, use_header=False)
    def __itruediv__(self, other):
        raise TypeError('can not modify protected units')

    @with_doc(Quantity.__ipow__, use_header=False)
    def __ipow__(self, other):
        raise TypeError('can not modify protected units')

    def __getstate__(self):
        """
        Return the internal state of the quantity, for pickling
        purposes.

        """
        state = (1, self._format_order)
        return state

    def __setstate__(self, state):
        ver, fo = state
        self._format_order = fo

    def __reduce__(self):
        """
        Return a tuple for pickling a UnitQuantity.
        """
        return (type(self), (self._name, self._definition, self._symbol, self._u_symbol, self._aliases, self.__doc__), self.__getstate__())

    def copy(self):
        return type(self)(self._name, self._definition, self._symbol, self._u_symbol, self._aliases, self.__doc__)