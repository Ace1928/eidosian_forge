from __future__ import (absolute_import, division, print_function)
from collections.abc import Mapping
from ansible import constants as C
from ansible.template import Templar, AnsibleUndefined

        Similar to __getitem__, however the returned data is not run through
        the templating engine to expand variables in the hostvars.
        