from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
@staticmethod
def url_encode(url_fragment):
    """
            replace special characters with URL encoding:
            %2F for /, %5C for backslash
        """
    return url_fragment.replace('/', '%2F').replace('\\', '%5C')