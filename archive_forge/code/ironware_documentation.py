from __future__ import (absolute_import, division, print_function)
import re
import json
from itertools import chain
from ansible.module_utils._text import to_bytes, to_text
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible.plugins.cliconf import CliconfBase, enable_mode

        Make sure we are in the operational cli mode
        :return: None
        