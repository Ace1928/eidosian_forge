import functools
import re
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win._i18n import _
from os_win import conf
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
def _set_switch_port_security_settings(self, switch_port_name, **kwargs):
    port_alloc = self._get_switch_port_allocation(switch_port_name)[0]
    sec_settings = self._get_security_setting_data_from_port_alloc(port_alloc)
    exists = sec_settings is not None
    if exists:
        if all((getattr(sec_settings, k) == v for k, v in kwargs.items())):
            return
    else:
        sec_settings = self._create_default_setting_data(self._PORT_SECURITY_SET_DATA)
    for k, v in kwargs.items():
        setattr(sec_settings, k, v)
    if exists:
        self._jobutils.modify_virt_feature(sec_settings)
    else:
        self._jobutils.add_virt_feature(sec_settings, port_alloc)
    sec_settings = self._get_security_setting_data_from_port_alloc(port_alloc)
    if not sec_settings:
        raise exceptions.HyperVException(_('Port Security Settings not found: %s') % switch_port_name)