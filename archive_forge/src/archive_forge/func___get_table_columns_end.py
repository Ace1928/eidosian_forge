from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb_canonical_map import base_interfaces
def __get_table_columns_end(headerline):
    """ fields length are diferent device to device, detect them on horizontal lin """
    fields_end = [m.start() for m in re.finditer('  *', headerline.strip())]
    fields_end.append(10000)
    return fields_end