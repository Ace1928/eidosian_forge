import sys
from pyasn1.type import error
class ConstraintsUnion(AbstractConstraintSet):
    """Create a ConstraintsUnion logic operator object.

    The ConstraintsUnion logic operator only succeeds if
    *at least a single* operand succeeds.

    The ConstraintsUnion object can be applied to
    any constraint and logic operator objects.

    The ConstraintsUnion object duck-types the immutable
    container object like Python :py:class:`tuple`.

    Parameters
    ----------
    \\*constraints:
        Constraint or logic operator objects.

    Examples
    --------
    .. code-block:: python

        class CapitalOrSmall(IA5String):
            '''
            ASN.1 specification:

            CapitalOrSmall ::=
                IA5String (FROM ("A".."Z") | FROM ("a".."z"))
            '''
            subtypeSpec = ConstraintsIntersection(
                PermittedAlphabetConstraint('A', 'Z'),
                PermittedAlphabetConstraint('a', 'z')
            )

        # this will succeed
        capital_or_small = CapitalAndSmall('Hello')

        # this will raise ValueConstraintError
        capital_or_small = CapitalOrSmall('hello!')
    """

    def _testValue(self, value, idx):
        for constraint in self._values:
            try:
                constraint(value, idx)
            except error.ValueConstraintError:
                pass
            else:
                return
        raise error.ValueConstraintError('all of %s failed for "%s"' % (self._values, value))