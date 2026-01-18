from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.xenserver import (xenserver_common_argument_spec, XenServerObject, get_object_ref,
Gathers and returns VM facts.