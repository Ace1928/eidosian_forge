import sys
from pyasn1.type import error
class ConstraintsIntersection(AbstractConstraintSet):
    """Create a ConstraintsIntersection logic operator object.

    The ConstraintsIntersection logic operator only succeeds
    if *all* its operands succeed.

    The ConstraintsIntersection object can be applied to
    any constraint and logic operator objects.

    The ConstraintsIntersection object duck-types the immutable
    container object like Python :py:class:`tuple`.

    Parameters
    ----------
    \\*constraints:
        Constraint or logic operator objects.

    Examples
    --------
    .. code-block:: python

        class CapitalAndSmall(IA5String):
            '''
            ASN.1 specification:

            CapitalAndSmall ::=
                IA5String (FROM ("A".."Z"|"a".."z"))
            '''
            subtypeSpec = ConstraintsIntersection(
                PermittedAlphabetConstraint('A', 'Z'),
                PermittedAlphabetConstraint('a', 'z')
            )

        # this will succeed
        capital_and_small = CapitalAndSmall('Hello')

        # this will raise ValueConstraintError
        capital_and_small = CapitalAndSmall('hello')
    """

    def _testValue(self, value, idx):
        for constraint in self._values:
            constraint(value, idx)