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
Removes the iApp tmplChecksum

        This is required for updating in place or else the load command will
        fail with a "AppTemplate ... content does not match the checksum"
        error.

        :return:
        