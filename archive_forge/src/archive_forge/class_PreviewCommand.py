from __future__ import (absolute_import, division, print_function)
import os
import json
from datetime import datetime
from os.path import exists
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import idrac_redfish_job_tracking, \
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.parse import urlparse
class PreviewCommand:

    def __init__(self, idrac, http_share, module):
        self.idrac = idrac
        self.http_share = http_share
        self.module = module

    def execute(self):
        scp_status = preview_scp_redfish(self.module, self.idrac, self.http_share, import_job_wait=False)
        return (scp_status, False)