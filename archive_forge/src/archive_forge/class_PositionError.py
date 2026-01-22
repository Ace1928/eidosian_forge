from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
class PositionError(FancyEqMixin):
    """
    Position error information.

    @cvar _ALLOWABLE_THRESHOLD: The maximum allowable difference between PDOP
        and the geometric mean of VDOP and HDOP. That difference is supposed
        to be zero, but can be non-zero because of rounding error and limited
        reporting precision. You should never have to change this value.
    @type _ALLOWABLE_THRESHOLD: C{float}
    @cvar _DOP_EXPRESSIONS: A mapping of DOP types (C[hvp]dop) to a list of
        callables that take self and return that DOP type, or raise
        C{TypeError}. This allows a DOP value to either be returned directly
        if it's know, or computed from other DOP types if it isn't.
    @type _DOP_EXPRESSIONS: C{dict} of C{str} to callables
    @ivar pdop: The position dilution of precision. L{None} if unknown.
    @type pdop: C{float} or L{None}
    @ivar hdop: The horizontal dilution of precision. L{None} if unknown.
    @type hdop: C{float} or L{None}
    @ivar vdop: The vertical dilution of precision. L{None} if unknown.
    @type vdop: C{float} or L{None}
    """
    compareAttributes = ('pdop', 'hdop', 'vdop')

    def __init__(self, pdop=None, hdop=None, vdop=None, testInvariant=False):
        """
        Initializes a positioning error object.

        @param pdop: The position dilution of precision. L{None} if unknown.
        @type pdop: C{float} or L{None}
        @param hdop: The horizontal dilution of precision. L{None} if unknown.
        @type hdop: C{float} or L{None}
        @param vdop: The vertical dilution of precision. L{None} if unknown.
        @type vdop: C{float} or L{None}
        @param testInvariant: Flag to test if the DOP invariant is valid or
            not. If C{True}, the invariant (PDOP = (HDOP**2 + VDOP**2)*.5) is
            checked at every mutation. By default, this is false, because the
            vast majority of DOP-providing devices ignore this invariant.
        @type testInvariant: c{bool}
        """
        self._pdop = pdop
        self._hdop = hdop
        self._vdop = vdop
        self._testInvariant = testInvariant
        self._testDilutionOfPositionInvariant()
    _ALLOWABLE_TRESHOLD = 0.01

    def _testDilutionOfPositionInvariant(self):
        """
        Tests if this positioning error object satisfies the dilution of
        position invariant (PDOP = (HDOP**2 + VDOP**2)*.5), unless the
        C{self._testInvariant} instance variable is C{False}.

        @return: L{None} if the invariant was not satisfied or not tested.
        @raises ValueError: Raised if the invariant was tested but not
            satisfied.
        """
        if not self._testInvariant:
            return
        for x in (self.pdop, self.hdop, self.vdop):
            if x is None:
                return
        delta = abs(self.pdop - (self.hdop ** 2 + self.vdop ** 2) ** 0.5)
        if delta > self._ALLOWABLE_TRESHOLD:
            raise ValueError('invalid combination of dilutions of precision: position: %s, horizontal: %s, vertical: %s' % (self.pdop, self.hdop, self.vdop))
    _DOP_EXPRESSIONS = {'pdop': [lambda self: float(self._pdop), lambda self: (self._hdop ** 2 + self._vdop ** 2) ** 0.5], 'hdop': [lambda self: float(self._hdop), lambda self: (self._pdop ** 2 - self._vdop ** 2) ** 0.5], 'vdop': [lambda self: float(self._vdop), lambda self: (self._pdop ** 2 - self._hdop ** 2) ** 0.5]}

    def _getDOP(self, dopType):
        """
        Gets a particular dilution of position value.

        @param dopType: The type of dilution of position to get. One of
            ('pdop', 'hdop', 'vdop').
        @type dopType: C{str}
        @return: The DOP if it is known, L{None} otherwise.
        @rtype: C{float} or L{None}
        """
        for dopExpression in self._DOP_EXPRESSIONS[dopType]:
            try:
                return dopExpression(self)
            except TypeError:
                continue

    def _setDOP(self, dopType, value):
        """
        Sets a particular dilution of position value.

        @param dopType: The type of dilution of position to set. One of
            ('pdop', 'hdop', 'vdop').
        @type dopType: C{str}

        @param value: The value to set the dilution of position type to.
        @type value: C{float}

        If this position error tests dilution of precision invariants,
        it will be checked. If the invariant is not satisfied, the
        assignment will be undone and C{ValueError} is raised.
        """
        attributeName = '_' + dopType
        oldValue = getattr(self, attributeName)
        setattr(self, attributeName, float(value))
        try:
            self._testDilutionOfPositionInvariant()
        except ValueError:
            setattr(self, attributeName, oldValue)
            raise

    @property
    def pdop(self):
        return self._getDOP('pdop')

    @pdop.setter
    def pdop(self, value):
        return self._setDOP('pdop', value)

    @property
    def hdop(self):
        return self._getDOP('hdop')

    @hdop.setter
    def hdop(self, value):
        return self._setDOP('hdop', value)

    @property
    def vdop(self):
        return self._getDOP('vdop')

    @vdop.setter
    def vdop(self, value):
        return self._setDOP('vdop', value)
    _REPR_TEMPLATE = '<PositionError (pdop: %s, hdop: %s, vdop: %s)>'

    def __repr__(self) -> str:
        """
        Returns a string representation of positioning information object.

        @return: The string representation.
        @rtype: C{str}
        """
        return self._REPR_TEMPLATE % (self.pdop, self.hdop, self.vdop)