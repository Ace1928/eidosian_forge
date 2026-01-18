from __future__ import (absolute_import, division, print_function)
import sys
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.facts.network.base import NetworkCollector

        Currently NVMe is only supported in some Linux distributions.
        If NVMe is configured on the host then a file will have been created
        during the NVMe driver installation. This file holds the unique NQN
        of the host.

        Example of contents of /etc/nvme/hostnqn:

        # cat /etc/nvme/hostnqn
        nqn.2014-08.org.nvmexpress:fc_lif:uuid:2cd61a74-17f9-4c22-b350-3020020c458d

        