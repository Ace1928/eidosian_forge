from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from datetime import datetime
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _monitors_and_quorum(self):
    if self.want.monitor_type is None:
        self.want.update(dict(monitor_type=self.have.monitor_type))
    if self.want.monitor_type == 'm_of_n':
        if self.want.quorum is None:
            self.want.update(dict(quorum=self.have.quorum))
        if self.want.quorum is None or self.want.quorum < 1:
            raise F5ModuleError("Quorum value must be specified with monitor_type 'm_of_n'.")
        if self.want.monitors != self.have.monitors:
            if self.want.monitors is None or not self.want.monitors_list:
                return None
            return dict(monitors=self.want.monitors)
    elif self.want.monitor_type == 'and_list':
        if self.want.quorum is not None and self.want.quorum > 0:
            raise F5ModuleError("Quorum values have no effect when used with 'and_list'.")
        if self.want.monitors != self.have.monitors:
            if self.want.monitors is None or not self.want.monitors_list:
                return None
            return dict(monitors=self.want.monitors)
    elif self.want.monitor_type == 'single':
        if len(self.want.monitors_list) > 1:
            raise F5ModuleError("When using a 'monitor_type' of 'single', only one monitor may be provided.")
        elif len(self.have.monitors_list) > 1 and len(self.want.monitors_list) == 0:
            raise F5ModuleError('A single monitor must be specified if more than one monitor currently exists on your pool.')
        self.want.update(dict(monitor_type='and_list'))
    if self.want.monitors != self.have.monitors:
        if self.want.monitors is None or not self.want.monitors_list:
            return None
        return dict(monitors=self.want.monitors)