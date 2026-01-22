from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
Reasonable attempt at validating a hostname

        Compiled from various paragraphs outlined here
        https://tools.ietf.org/html/rfc3696#section-2
        https://tools.ietf.org/html/rfc1123

        Notably,
        * Host software MUST handle host names of up to 63 characters and
          SHOULD handle host names of up to 255 characters.
        * The "LDH rule", after the characters that it permits. (letters, digits, hyphen)
        * If the hyphen is used, it is not permitted to appear at
          either the beginning or end of a label

        :param host:
        :return:
        