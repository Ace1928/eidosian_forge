import enum
def _error_code_from_int(code):
    """
    Given an integer error code, returns either one of :class:`ErrorCodes
    <h2.errors.ErrorCodes>` or, if not present in the known set of codes,
    returns the integer directly.
    """
    try:
        return ErrorCodes(code)
    except ValueError:
        return code