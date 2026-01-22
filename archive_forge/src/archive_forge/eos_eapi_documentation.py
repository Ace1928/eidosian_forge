from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.arista.eos.plugins.module_utils.network.eos.eos import (
main entry point for module execution