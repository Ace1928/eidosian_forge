import re
def uplowcase(string, case):
    """Convert string into upper or lower case.

    Args:
        string: String to convert.

    Returns:
        string: Uppercase or lowercase case string.

    """
    if case == 'up':
        return str(string).upper()
    elif case == 'low':
        return str(string).lower()