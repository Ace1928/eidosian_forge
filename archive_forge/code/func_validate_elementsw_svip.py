from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def validate_elementsw_svip(self, path, elem):
    """
        Validate ElementSW cluster SVIP
        :return: None
        """
    result = None
    try:
        result = elem.get_cluster_info()
    except solidfire.common.ApiServerError as err:
        self.module.fail_json(msg='Error fetching SVIP', exception=to_native(err))
    if result and result.cluster_info.svip:
        cluster_svip = result.cluster_info.svip
        svip = path.split(':')[0]
        if svip != cluster_svip:
            self.module.fail_json(msg='Error: Invalid SVIP')