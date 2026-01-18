from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_native
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
Return the possibly of a file being consumable by this plugin.