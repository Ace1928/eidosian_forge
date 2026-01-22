class NumericalVerifyError(VerifyErrorBase):
    """
    The base for all exceptions resulting from a failed numerical
    verification of an equality (using some epsilon) or inequality
    (typically by interval arithmetics).
    """