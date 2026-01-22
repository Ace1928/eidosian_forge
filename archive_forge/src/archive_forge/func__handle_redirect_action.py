from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _handle_redirect_action(self, action, item):
    """Handle the nuances of the redirect type

        :param action:
        :param item:
        :return:
        """
    action['type'] = 'redirect'
    if 'location' not in item:
        raise F5ModuleError("A 'location' must be specified when the 'redirect' type is used.")
    action.update(location=item['location'], httpReply=True)