from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils._text import to_native, to_text
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, ConfigLine
def voss_parse(lines, indent=None, comment_tokens=None):
    toplevel = re.compile('(^interface.*$)|(^router \\w+$)|(^router vrf \\w+$)')
    exitline = re.compile('^exit$')
    entry_reg = re.compile('([{};])')
    ancestors = list()
    config = list()
    dup_parent_index = None
    for line in to_native(lines, errors='surrogate_or_strict').split('\n'):
        text = entry_reg.sub('', line).strip()
        cfg = ConfigLine(text)
        if not text or ignore_line(text, comment_tokens):
            continue
        if toplevel.match(text):
            for index, item in enumerate(config):
                if item.text == text:
                    dup_parent_index = index
                    break
            ancestors = [cfg]
            config.append(cfg)
        elif exitline.match(text):
            ancestors = list()
            if dup_parent_index is not None:
                dup_parent_index = None
            else:
                cfg._parents = ancestors[:1]
                config.append(cfg)
        elif ancestors:
            cfg._parents = ancestors[:1]
            if dup_parent_index is not None:
                config[int(dup_parent_index)].add_child(cfg)
                new_index = dup_parent_index + 1
                config.insert(new_index, cfg)
            else:
                ancestors[0].add_child(cfg)
                config.append(cfg)
        else:
            config.append(cfg)
    return config