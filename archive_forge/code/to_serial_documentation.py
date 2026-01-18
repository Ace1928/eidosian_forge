from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import integer_types
from ansible_collections.community.crypto.plugins.module_utils.serial import to_serial
Ansible jinja2 filters