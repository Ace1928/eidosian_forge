import sys
from pyasn1.type import error
class AbstractConstraint(object):

    def __init__(self, *values):
        self._valueMap = set()
        self._setValues(values)
        self.__hash = hash((self.__class__.__name__, self._values))

    def __call__(self, value, idx=None):
        if not self._values:
            return
        try:
            self._testValue(value, idx)
        except error.ValueConstraintError:
            raise error.ValueConstraintError('%s failed at: %r' % (self, sys.exc_info()[1]))

    def __repr__(self):
        representation = '%s object at 0x%x' % (self.__class__.__name__, id(self))
        if self._values:
            representation += ' consts %s' % ', '.join([repr(x) for x in self._values])
        return '<%s>' % representation

    def __eq__(self, other):
        return self is other and True or self._values == other

    def __ne__(self, other):
        return self._values != other

    def __lt__(self, other):
        return self._values < other

    def __le__(self, other):
        return self._values <= other

    def __gt__(self, other):
        return self._values > other

    def __ge__(self, other):
        return self._values >= other
    if sys.version_info[0] <= 2:

        def __nonzero__(self):
            return self._values and True or False
    else:

        def __bool__(self):
            return self._values and True or False

    def __hash__(self):
        return self.__hash

    def _setValues(self, values):
        self._values = values

    def _testValue(self, value, idx):
        raise error.ValueConstraintError(value)

    def getValueMap(self):
        return self._valueMap

    def isSuperTypeOf(self, otherConstraint):
        return otherConstraint is self or not self._values or otherConstraint == self or (self in otherConstraint.getValueMap())

    def isSubTypeOf(self, otherConstraint):
        return otherConstraint is self or not self or otherConstraint == self or (otherConstraint in self._valueMap)