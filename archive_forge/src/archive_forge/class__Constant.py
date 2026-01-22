from __future__ import division, absolute_import
from functools import partial
from itertools import count
from operator import and_, or_, xor
class _Constant(object):
    """
    @ivar _index: A C{int} allocated from a shared counter in order to keep
        track of the order in which L{_Constant}s are instantiated.

    @ivar name: A C{str} giving the name of this constant; only set once the
        constant is initialized by L{_ConstantsContainer}.

    @ivar _container: The L{_ConstantsContainer} subclass this constant belongs
        to; C{None} until the constant is initialized by that subclass.
    """

    def __init__(self):
        self._container = None
        self._index = _constantOrder()

    def __repr__(self):
        """
        Return text identifying both which constant this is and which
        collection it belongs to.
        """
        return '<%s=%s>' % (self._container.__name__, self.name)

    def __lt__(self, other):
        """
        Implements C{<}.  Order is defined by instantiation order.

        @param other: An object.

        @return: C{NotImplemented} if C{other} is not a constant belonging to
            the same container as this constant, C{True} if this constant is
            defined before C{other}, otherwise C{False}.
        """
        if not isinstance(other, self.__class__) or not self._container == other._container:
            return NotImplemented
        return self._index < other._index

    def __le__(self, other):
        """
        Implements C{<=}.  Order is defined by instantiation order.

        @param other: An object.

        @return: C{NotImplemented} if C{other} is not a constant belonging to
            the same container as this constant, C{True} if this constant is
            defined before or equal to C{other}, otherwise C{False}.
        """
        if not isinstance(other, self.__class__) or not self._container == other._container:
            return NotImplemented
        return self is other or self._index < other._index

    def __gt__(self, other):
        """
        Implements C{>}.  Order is defined by instantiation order.

        @param other: An object.

        @return: C{NotImplemented} if C{other} is not a constant belonging to
            the same container as this constant, C{True} if this constant is
            defined after C{other}, otherwise C{False}.
        """
        if not isinstance(other, self.__class__) or not self._container == other._container:
            return NotImplemented
        return self._index > other._index

    def __ge__(self, other):
        """
        Implements C{>=}.  Order is defined by instantiation order.

        @param other: An object.

        @return: C{NotImplemented} if C{other} is not a constant belonging to
            the same container as this constant, C{True} if this constant is
            defined after or equal to C{other}, otherwise C{False}.
        """
        if not isinstance(other, self.__class__) or not self._container == other._container:
            return NotImplemented
        return self is other or self._index > other._index

    def _realize(self, container, name, value):
        """
        Complete the initialization of this L{_Constant}.

        @param container: The L{_ConstantsContainer} subclass this constant is
            part of.

        @param name: The name of this constant in its container.

        @param value: The value of this constant; not used, as named constants
            have no value apart from their identity.
        """
        self._container = container
        self.name = name