from __future__ import absolute_import, division, print_function
import copy
import os
import re
import datetime
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import exec_command
from ansible.module_utils.six import iteritems
from ansible.module_utils.parsing.convert_bool import (
from collections import defaultdict
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from .constants import (
def api_params(self):
    result = {}
    for api_attribute in self.api_attributes:
        if self.api_map is not None and api_attribute in self.api_map:
            result[api_attribute] = getattr(self, self.api_map[api_attribute])
        else:
            result[api_attribute] = getattr(self, api_attribute)
    result = self._filter_params(result)
    return result