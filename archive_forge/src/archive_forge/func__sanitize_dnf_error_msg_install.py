from __future__ import absolute_import, division, print_function
import os
import re
import sys
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_file
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
def _sanitize_dnf_error_msg_install(self, spec, error):
    """
        For unhandled dnf.exceptions.Error scenarios, there are certain error
        messages we want to filter in an install scenario. Do that here.
        """
    if to_text('no package matched') in to_text(error) or to_text('No match for argument:') in to_text(error):
        return 'No package {0} available.'.format(spec)
    return error