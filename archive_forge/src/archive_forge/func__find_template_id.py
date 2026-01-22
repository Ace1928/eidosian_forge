from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _find_template_id(module, datacenter):
    """
        Find the template id by calling the CLC API.
        :param module: the module to validate
        :param datacenter: the datacenter to search for the template
        :return: a valid clc template id
        """
    lookup_template = module.params.get('template')
    state = module.params.get('state')
    type = module.params.get('type')
    result = None
    if state == 'present' and type != 'bareMetal':
        try:
            result = datacenter.Templates().Search(lookup_template)[0].id
        except CLCException:
            module.fail_json(msg=str('Unable to find a template: ' + lookup_template + ' in location: ' + datacenter.id))
    return result