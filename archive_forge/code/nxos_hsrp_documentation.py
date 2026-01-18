from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
Some older NXOS images fail to set the attr values when using structured output and
    instead set the values to <unknown enum>. This fallback method is a workaround that
    uses an unstructured (text) request to query the device a second time.
    'sh_preempt' is currently the only attr affected. Add checks for other attrs as needed.
    