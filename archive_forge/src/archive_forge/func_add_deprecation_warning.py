from functools import partial
from modin.utils import align_indents, append_to_docstring, format_string
def add_deprecation_warning(replacement_method):
    """
    Build decorator which appends deprecation warning to the function's docstring.

    Appended warning indicates that the current method duplicates functionality of
    some other method and so is slated to be removed in the future.

    Parameters
    ----------
    replacement_method : str
        Name of the method to use instead of deprecated.

    Returns
    -------
    callable
    """
    message = _deprecation_warning.format(replacement_method)
    return append_to_docstring(message)