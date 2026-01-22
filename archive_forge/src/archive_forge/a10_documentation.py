from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.urls import fetch_url

    The axapi uses 0/1 integer values for flags, rather than strings
    or booleans, so convert the given flag to a 0 or 1. For now, params
    are specified as strings only so thats what we check.
    