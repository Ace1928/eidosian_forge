from __future__ import (absolute_import, division, print_function)
import re
from ansible.errors import AnsibleFilterError
def to_minutes(human_time, **kwargs):
    """ Return minutes from a human readable string """
    return to_time_unit(human_time, 'm', **kwargs)