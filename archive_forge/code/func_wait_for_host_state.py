from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.opennebula import OpenNebulaModule
def wait_for_host_state(self, host, target_states):
    """
        Utility method that waits for a host state.
        Args:
            host:
            target_states:

        """
    return self.wait_for_state('host', lambda: self.one.host.info(host.ID).STATE, lambda s: HOST_STATES(s).name, target_states, invalid_states=[HOST_STATES.ERROR, HOST_STATES.MONITORING_ERROR])