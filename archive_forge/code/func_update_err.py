from __future__ import absolute_import, division, print_function
import itertools
from ansible.errors import AnsibleFilterError
def update_err(*args, **kwargs):
    """Filters return value or raises error as per supplied parameters

        Returns:
            any: Return value to the function call
        """
    res, err = validator_func(*args, **kwargs)
    if any([err.get('fail_missing_match_key'), err.get('fail_duplicate'), err.get('fail_missing_match_value')]):
        _raise_error(err)
    return res