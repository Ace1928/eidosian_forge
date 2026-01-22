from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
def _set_default_creation_values(self):
    if self.want.timeout is None:
        self.want.update({'timeout': 120})
    if self.want.interval is None:
        self.want.update({'interval': 30})
    if self.want.ip is None:
        self.want.update({'ip': '*'})
    if self.want.port is None:
        self.want.update({'port': '*'})
    if self.want.probe_interval is None:
        self.want.update({'probe_interval': 1})
    if self.want.probe_timeout is None:
        self.want.update({'probe_timeout': 5})
    if self.want.probe_attempts is None:
        self.want.update({'probe_attempts': 3})
    if self.want.ignore_down_response is None:
        self.want.update({'ignore_down_response': False})
    if self.want.transparent is None:
        self.want.update({'transparent': False})