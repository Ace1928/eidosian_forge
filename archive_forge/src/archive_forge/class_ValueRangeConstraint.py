import sys
from pyasn1.type import error
class ValueRangeConstraint(AbstractConstraint):
    """Create a ValueRangeConstraint object.

    The ValueRangeConstraint satisfies any value that
    falls in the range of permitted values.

    The ValueRangeConstraint object can only be applied
    to :class:`~pyasn1.type.univ.Integer` and
    :class:`~pyasn1.type.univ.Real` types.

    Parameters
    ----------
    start: :class:`int`
        Minimum permitted value in the range (inclusive)

    end: :class:`int`
        Maximum permitted value in the range (inclusive)

    Examples
    --------
    .. code-block:: python

        class TeenAgeYears(Integer):
            '''
            ASN.1 specification:

            TeenAgeYears ::= INTEGER (13 .. 19)
            '''
            subtypeSpec = ValueRangeConstraint(13, 19)

        # this will succeed
        teen_year = TeenAgeYears(18)

        # this will raise ValueConstraintError
        teen_year = TeenAgeYears(20)
    """

    def _testValue(self, value, idx):
        if value < self.start or value > self.stop:
            raise error.ValueConstraintError(value)

    def _setValues(self, values):
        if len(values) != 2:
            raise error.PyAsn1Error('%s: bad constraint values' % (self.__class__.__name__,))
        self.start, self.stop = values
        if self.start > self.stop:
            raise error.PyAsn1Error('%s: screwed constraint values (start > stop): %s > %s' % (self.__class__.__name__, self.start, self.stop))
        AbstractConstraint._setValues(self, values)