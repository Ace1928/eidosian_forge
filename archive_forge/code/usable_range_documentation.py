from __future__ import absolute_import, division, print_function
from ipaddress import IPv4Network, IPv6Network
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.utils import _validate_args
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import ensure_text
a mapping of filter names to functions