import os
import fnmatch
def _filter_names(names):
    """
    Given a list of file names, return those names that should be copied.
    """
    names = [n for n in names if n not in EXCLUDE_NAMES]
    for pattern in EXCLUDE_PATTERNS:
        names = [n for n in names if not fnmatch.fnmatch(n, pattern) and (not n.endswith('.py'))]
    return names