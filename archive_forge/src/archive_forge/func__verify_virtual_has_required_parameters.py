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
def _verify_virtual_has_required_parameters(self):
    """Verify that the virtual has required parameters

        Virtual servers require several parameters that are not necessarily required
        when updating the virtual. This method will check for the required params
        upon creation.

        Ansible supports ``default`` variables in an Argument Spec, but those defaults
        apply to all operations; including create, update, and delete. Since users are not
        required to always specify these parameters, we cannot use Ansible's facility.
        If we did, and then users would be required to provide them when, for example,
        they attempted to delete a virtual (even though they are not required to delete
        a virtual.

        Raises:
             F5ModuleError: Raised when the user did not specify required parameters.
        """
    required_resources = ['destination', 'port']
    if self.want.type == 'internal':
        return
    if all((getattr(self.want, v) is None for v in required_resources)):
        raise F5ModuleError('You must specify both of ' + ', '.join(required_resources))