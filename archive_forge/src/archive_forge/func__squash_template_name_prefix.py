from __future__ import absolute_import, division, print_function
import re
import uuid
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _squash_template_name_prefix(self):
    """Removes the template name prefix

        This method removes that partition from the name
        in the iApp so that comparisons can be done properly and entries
        can be created properly when using REST.

        :return string
        """
    pattern = 'sys\\s+application\\s+template\\s+/Common/'
    replace = 'sys application template '
    return re.sub(pattern, replace, self._values['content'])