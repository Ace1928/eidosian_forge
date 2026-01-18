from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes
def update_existing_option_line(target_option, value):
    old_line = target_option['line']
    old_value = target_option['value']
    prefix_start = old_line.find(target_option['option'])
    optionLen = len(target_option['option'])
    old_value_position = re.search('\\s+'.join(map(re.escape, old_value.split())), old_line[prefix_start + optionLen:])
    start = old_value_position.start() + prefix_start + optionLen
    end = old_value_position.end() + prefix_start + optionLen
    line = old_line[:start] + value + old_line[end:]
    return line