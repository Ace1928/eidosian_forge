from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule

    By convention if a module is loaded via name, it appears in apache2ctl -M as
    name_module.

    Some modules don't follow this convention and we use replacements for those.