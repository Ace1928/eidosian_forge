class TiltIsZeroExactVerifyError(IsZeroExactVerifyError, TiltType):
    """
    Verifying that a tilt is zero has failed using exact arithmetic.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'Verification that tilt is zero has failed using exact arithmetic: %r == 0' % self.value