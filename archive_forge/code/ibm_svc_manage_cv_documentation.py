from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native

        find the remote copy relationships such as Metro Mirror, Global Mirror
        relationships visible to the system.

        Returns:
            None if no matching instances or a list including all the matching
            instances
        