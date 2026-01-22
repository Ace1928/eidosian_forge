class TiltInequalityNumericalVerifyError(InequalityNumericalVerifyError, TiltType):
    """
    Numerically verifying that a tilt is negative has failed.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'Numerical verification that tilt is negative has failed: %r < 0' % self.value