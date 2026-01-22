from __future__ import absolute_import, division, print_function
import hashlib
import os
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
Return SHA1 checksum of the file on disk

        Returns:
            string: The SHA1 checksum of the file.

        References:
            - https://stackoverflow.com/a/22058673/661215
        