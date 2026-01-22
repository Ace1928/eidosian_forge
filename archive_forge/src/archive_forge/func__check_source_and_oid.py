from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _check_source_and_oid(self):
    if self.have.oid is not None:
        if self.want.source == 'all' and self.want.oid != '':
            raise F5ModuleError("When specifying an 'all' source for a resource with an existing OID, you must specify a new, empty, OID.")
    if self.want.source == 'all' and self.want.oid != '':
        raise F5ModuleError("When specifying an 'all' source for a resource, you may not specify an OID.")