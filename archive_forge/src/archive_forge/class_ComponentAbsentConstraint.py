import sys
from pyasn1.type import error
class ComponentAbsentConstraint(AbstractConstraint):
    """Create a ComponentAbsentConstraint object.

    The ComponentAbsentConstraint is only satisfied when the value
    is `None`.

    The ComponentAbsentConstraint object is typically used with
    `WithComponentsConstraint`.

    Examples
    --------
    .. code-block:: python

        absent = ComponentAbsentConstraint()

        # this will succeed
        absent(None)

        # this will raise ValueConstraintError
        absent('whatever')
    """

    def _setValues(self, values):
        self._values = ('<must be absent>',)
        if values:
            raise error.PyAsn1Error('No arguments expected')

    def _testValue(self, value, idx):
        if value is not None:
            raise error.ValueConstraintError('Component is not absent: %r' % value)