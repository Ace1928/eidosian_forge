from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb_canonical_map import base_interfaces
def __line_to_fields(line, fields_end):
    """ dynamic fields lenghts """
    line_elems = {}
    index = 0
    f_start = 0
    for f_end in fields_end:
        line_elems[index] = line[f_start:f_end].strip()
        index += 1
        f_start = f_end
    return line_elems