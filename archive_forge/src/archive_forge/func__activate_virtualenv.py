from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import rax_argument_spec, rax_clb_node_to_dict, rax_required_together, setup_rax_module
def _activate_virtualenv(path):
    activate_this = os.path.join(path, 'bin', 'activate_this.py')
    with open(activate_this) as f:
        code = compile(f.read(), activate_this, 'exec')
        exec(code)