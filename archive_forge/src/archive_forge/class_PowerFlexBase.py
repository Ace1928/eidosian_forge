from __future__ import (absolute_import, division, print_function)
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
class PowerFlexBase:
    """PowerFlex Base Class"""

    def __init__(self, ansible_module, ansible_module_params):
        """
        Initialize the powerflex base class

        :param ansible_module: Ansible module class
        :type ansible_module: AnsibleModule
        :param ansible_module_params: Parameters for ansible module class
        :type ansible_module_params: dict
        """
        self.module_params = utils.get_powerflex_gateway_host_parameters()
        ansible_module_params['argument_spec'].update(self.module_params)
        self.module = ansible_module(**ansible_module_params)
        utils.ensure_required_libs(self.module)
        self.result = {'changed': False}
        try:
            self.powerflex_conn = utils.get_powerflex_gateway_host_connection(self.module.params)
            LOG.info('Got the PowerFlex system connection object instance')
        except Exception as e:
            LOG.error(str(e))
            self.module.fail_json(msg=str(e))