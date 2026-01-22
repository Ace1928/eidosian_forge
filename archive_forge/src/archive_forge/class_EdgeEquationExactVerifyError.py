class EdgeEquationExactVerifyError(ExactVerifyError, EdgeEquationType):
    """
    Exception for failed verification of a polynomial edge equation
    using exact arithmetics.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'Verification of a polynomial edge equation using exact arithmetic failed: %r == 1' % self.value