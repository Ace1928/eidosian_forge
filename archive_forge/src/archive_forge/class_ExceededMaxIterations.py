class ExceededMaxIterations(NetworkXException):
    """Raised if a loop iterates too many times without breaking.

    This may occur, for example, in an algorithm that computes
    progressively better approximations to a value but exceeds an
    iteration bound specified by the user.

    """