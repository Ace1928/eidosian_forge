import sys
from pyasn1.type import error
class ConstraintsExclusion(AbstractConstraint):
    """Create a ConstraintsExclusion logic operator object.

    The ConstraintsExclusion logic operator succeeds when the
    value does *not* satisfy the operand constraint.

    The ConstraintsExclusion object can be applied to
    any constraint and logic operator object.

    Parameters
    ----------
    constraint:
        Constraint or logic operator object.

    Examples
    --------
    .. code-block:: python

        class Lipogramme(IA5STRING):
            '''
            ASN.1 specification:

            Lipogramme ::=
                IA5String (FROM (ALL EXCEPT ("e"|"E")))
            '''
            subtypeSpec = ConstraintsExclusion(
                PermittedAlphabetConstraint('e', 'E')
            )

        # this will succeed
        lipogramme = Lipogramme('A work of fiction?')

        # this will raise ValueConstraintError
        lipogramme = Lipogramme('Eel')

    Warning
    -------
    The above example involving PermittedAlphabetConstraint might
    not work due to the way how PermittedAlphabetConstraint works.
    The other constraints might work with ConstraintsExclusion
    though.
    """

    def _testValue(self, value, idx):
        try:
            self._values[0](value, idx)
        except error.ValueConstraintError:
            return
        else:
            raise error.ValueConstraintError(value)

    def _setValues(self, values):
        if len(values) != 1:
            raise error.PyAsn1Error('Single constraint expected')
        AbstractConstraint._setValues(self, values)