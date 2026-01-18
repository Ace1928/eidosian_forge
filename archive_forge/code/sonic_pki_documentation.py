from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.pki.pki import PkiArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.config.pki.pki import Pki

    Main entry point for module execution

    :returns: the result form module invocation
    