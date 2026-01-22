from __future__ import (absolute_import, division, print_function)
import sys
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.facts.network.base import NetworkCollector
class NvmeInitiatorNetworkCollector(NetworkCollector):
    name = 'nvme'
    _fact_ids = set()

    def collect(self, module=None, collected_facts=None):
        """
        Currently NVMe is only supported in some Linux distributions.
        If NVMe is configured on the host then a file will have been created
        during the NVMe driver installation. This file holds the unique NQN
        of the host.

        Example of contents of /etc/nvme/hostnqn:

        # cat /etc/nvme/hostnqn
        nqn.2014-08.org.nvmexpress:fc_lif:uuid:2cd61a74-17f9-4c22-b350-3020020c458d

        """
        nvme_facts = {}
        nvme_facts['hostnqn'] = ''
        if sys.platform.startswith('linux'):
            for line in get_file_content('/etc/nvme/hostnqn', '').splitlines():
                if line.startswith('#') or line.startswith(';') or line.strip() == '':
                    continue
                if line.startswith('nqn.'):
                    nvme_facts['hostnqn'] = line
                    break
        return nvme_facts