from __future__ import absolute_import, division, print_function
import copy
from ansible.errors import AnsibleError
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def _get_implementation_module(self, network_os, platform_agnostic_module):
    module_name = network_os.split('.')[-1] + '_' + platform_agnostic_module.partition('_')[2]
    if '.' in network_os:
        fqcn_module = '.'.join(network_os.split('.')[0:-1])
        implementation_module = fqcn_module + '.' + module_name
    else:
        implementation_module = module_name
    if implementation_module not in self._shared_loader_obj.module_loader:
        implementation_module = None
    return implementation_module