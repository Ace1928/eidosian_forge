import warnings
from warnings import warn
import breezy
def _check_for_filter(error_only):
    """Check if there is already a filter for deprecation warnings.

    :param error_only: Only match an 'error' filter
    :return: True if a filter is found, False otherwise
    """
    for filter in warnings.filters:
        if issubclass(DeprecationWarning, filter[2]):
            if not error_only or filter[0] == 'error':
                return True
    return False