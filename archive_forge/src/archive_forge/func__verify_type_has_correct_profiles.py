from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.constants import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
def _verify_type_has_correct_profiles(self):
    """Verify that specified server type does not include forbidden profiles

        The type of the server determines the ``type``s of profiles that it accepts. This
        method checks that the server ``type`` that you specified is indeed one that can
        accept the profiles that you specified.

        The common situations are

        * ``standard`` types that include ``fasthttp``, ``fastl4``, or ``message routing`` profiles
        * ``fasthttp`` types that are missing a ``fasthttp`` profile
        * ``fastl4`` types that are missing a ``fastl4`` profile
        * ``message-routing`` types that are missing ``diameter`` or ``sip`` profiles

        Raises:
            F5ModuleError: Raised when a validation check fails.
        """
    if self.want.type == 'standard':
        if self.want.has_fasthttp_profiles:
            raise F5ModuleError("A 'standard' type may not have 'fasthttp' profiles.")
        if self.want.has_fastl4_profiles:
            raise F5ModuleError("A 'standard' type may not have 'fastl4' profiles.")
    elif self.want.type == 'performance-http':
        if not self.want.has_fasthttp_profiles:
            raise F5ModuleError("A 'fasthttp' type must have at least one 'fasthttp' profile.")
    elif self.want.type == 'performance-l4':
        if not self.want.has_fastl4_profiles:
            raise F5ModuleError("A 'fastl4' type must have at least one 'fastl4' profile.")
    elif self.want.type == 'message-routing':
        if not self.want.has_message_routing_profiles:
            raise F5ModuleError("A 'message-routing' type must have either a 'sip' or 'diameter' profile.")