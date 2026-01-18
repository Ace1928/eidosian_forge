from __future__ import (absolute_import, division, print_function)
import re
from ansible.errors import AnsibleFilterError
def to_years(human_time, **kwargs):
    """ Return years from a human readable string """
    return to_time_unit(human_time, 'y', **kwargs)