from __future__ import (absolute_import, division, print_function)
import copy
import json
import os
import re
import traceback
from io import BytesIO
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, json_dict_bytes_to_unicode, missing_required_lib
from ansible.module_utils.six import iteritems, string_types
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.common._collections_compat import MutableMapping
def add_target_children(module, tree, xpath, namespaces, children, in_type, insertbefore, insertafter):
    if is_node(tree, xpath, namespaces):
        new_kids = children_to_nodes(module, children, in_type)
        if insertbefore or insertafter:
            insert_target_children(tree, xpath, namespaces, new_kids, insertbefore, insertafter)
        else:
            for node in tree.xpath(xpath, namespaces=namespaces):
                node.extend(new_kids)
        finish(module, tree, xpath, namespaces, changed=True)
    else:
        finish(module, tree, xpath, namespaces)