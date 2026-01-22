import math
import sys
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import integer
from pyasn1.compat import octets
from pyasn1.type import base
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import tagmap
class Real(base.SimpleAsn1Type):
    """Create |ASN.1| schema or value object.

    |ASN.1| class is based on :class:`~pyasn1.type.base.SimpleAsn1Type`, its
    objects are immutable and duck-type Python :class:`float` objects.
    Additionally, |ASN.1| objects behave like a :class:`tuple` in which case its
    elements are mantissa, base and exponent.

    Keyword Args
    ------------
    value: :class:`tuple`, :class:`float` or |ASN.1| object
        Python sequence of :class:`int` (representing mantissa, base and
        exponent) or :class:`float` instance or |ASN.1| object.
        If `value` is not given, schema object will be created.

    tagSet: :py:class:`~pyasn1.type.tag.TagSet`
        Object representing non-default ASN.1 tag(s)

    subtypeSpec: :py:class:`~pyasn1.type.constraint.ConstraintsIntersection`
        Object representing non-default ASN.1 subtype constraint(s). Constraints
        verification for |ASN.1| type occurs automatically on object
        instantiation.

    Raises
    ------
    ~pyasn1.error.ValueConstraintError, ~pyasn1.error.PyAsn1Error
        On constraint violation or bad initializer.

    Examples
    --------
    .. code-block:: python

        class Pi(Real):
            '''
            ASN.1 specification:

            Pi ::= REAL

            pi Pi ::= { mantissa 314159, base 10, exponent -5 }

            '''
        pi = Pi((314159, 10, -5))
    """
    binEncBase = None
    try:
        _plusInf = float('inf')
        _minusInf = float('-inf')
        _inf = (_plusInf, _minusInf)
    except ValueError:
        _plusInf = _minusInf = None
        _inf = ()
    tagSet = tag.initTagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 9))
    subtypeSpec = constraint.ConstraintsIntersection()
    typeId = base.SimpleAsn1Type.getTypeId()

    @staticmethod
    def __normalizeBase10(value):
        m, b, e = value
        while m and m % 10 == 0:
            m /= 10
            e += 1
        return (m, b, e)

    def prettyIn(self, value):
        if isinstance(value, tuple) and len(value) == 3:
            if not isinstance(value[0], numericTypes) or not isinstance(value[1], intTypes) or (not isinstance(value[2], intTypes)):
                raise error.PyAsn1Error('Lame Real value syntax: %s' % (value,))
            if isinstance(value[0], float) and self._inf and (value[0] in self._inf):
                return value[0]
            if value[1] not in (2, 10):
                raise error.PyAsn1Error('Prohibited base for Real value: %s' % (value[1],))
            if value[1] == 10:
                value = self.__normalizeBase10(value)
            return value
        elif isinstance(value, intTypes):
            return self.__normalizeBase10((value, 10, 0))
        elif isinstance(value, float) or octets.isStringType(value):
            if octets.isStringType(value):
                try:
                    value = float(value)
                except ValueError:
                    raise error.PyAsn1Error('Bad real value syntax: %s' % (value,))
            if self._inf and value in self._inf:
                return value
            else:
                e = 0
                while int(value) != value:
                    value *= 10
                    e -= 1
                return self.__normalizeBase10((int(value), 10, e))
        elif isinstance(value, Real):
            return tuple(value)
        raise error.PyAsn1Error('Bad real value syntax: %s' % (value,))

    def prettyPrint(self, scope=0):
        try:
            return self.prettyOut(float(self))
        except OverflowError:
            return '<overflow>'

    @property
    def isPlusInf(self):
        """Indicate PLUS-INFINITY object value

        Returns
        -------
        : :class:`bool`
            :obj:`True` if calling object represents plus infinity
            or :obj:`False` otherwise.

        """
        return self._value == self._plusInf

    @property
    def isMinusInf(self):
        """Indicate MINUS-INFINITY object value

        Returns
        -------
        : :class:`bool`
            :obj:`True` if calling object represents minus infinity
            or :obj:`False` otherwise.
        """
        return self._value == self._minusInf

    @property
    def isInf(self):
        return self._value in self._inf

    def __add__(self, value):
        return self.clone(float(self) + value)

    def __radd__(self, value):
        return self + value

    def __mul__(self, value):
        return self.clone(float(self) * value)

    def __rmul__(self, value):
        return self * value

    def __sub__(self, value):
        return self.clone(float(self) - value)

    def __rsub__(self, value):
        return self.clone(value - float(self))

    def __mod__(self, value):
        return self.clone(float(self) % value)

    def __rmod__(self, value):
        return self.clone(value % float(self))

    def __pow__(self, value, modulo=None):
        return self.clone(pow(float(self), value, modulo))

    def __rpow__(self, value):
        return self.clone(pow(value, float(self)))
    if sys.version_info[0] <= 2:

        def __div__(self, value):
            return self.clone(float(self) / value)

        def __rdiv__(self, value):
            return self.clone(value / float(self))
    else:

        def __truediv__(self, value):
            return self.clone(float(self) / value)

        def __rtruediv__(self, value):
            return self.clone(value / float(self))

        def __divmod__(self, value):
            return self.clone(float(self) // value)

        def __rdivmod__(self, value):
            return self.clone(value // float(self))

    def __int__(self):
        return int(float(self))
    if sys.version_info[0] <= 2:

        def __long__(self):
            return long(float(self))

    def __float__(self):
        if self._value in self._inf:
            return self._value
        else:
            return float(self._value[0] * pow(self._value[1], self._value[2]))

    def __abs__(self):
        return self.clone(abs(float(self)))

    def __pos__(self):
        return self.clone(+float(self))

    def __neg__(self):
        return self.clone(-float(self))

    def __round__(self, n=0):
        r = round(float(self), n)
        if n:
            return self.clone(r)
        else:
            return r

    def __floor__(self):
        return self.clone(math.floor(float(self)))

    def __ceil__(self):
        return self.clone(math.ceil(float(self)))

    def __trunc__(self):
        return self.clone(math.trunc(float(self)))

    def __lt__(self, value):
        return float(self) < value

    def __le__(self, value):
        return float(self) <= value

    def __eq__(self, value):
        return float(self) == value

    def __ne__(self, value):
        return float(self) != value

    def __gt__(self, value):
        return float(self) > value

    def __ge__(self, value):
        return float(self) >= value
    if sys.version_info[0] <= 2:

        def __nonzero__(self):
            return bool(float(self))
    else:

        def __bool__(self):
            return bool(float(self))
        __hash__ = base.SimpleAsn1Type.__hash__

    def __getitem__(self, idx):
        if self._value in self._inf:
            raise error.PyAsn1Error('Invalid infinite value operation')
        else:
            return self._value[idx]

    def isPlusInfinity(self):
        return self.isPlusInf

    def isMinusInfinity(self):
        return self.isMinusInf

    def isInfinity(self):
        return self.isInf