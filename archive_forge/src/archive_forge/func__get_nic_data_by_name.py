import functools
import time
import uuid
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import uuidutils
from six.moves import range  # noqa
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
from os_win.utils import pathutils
def _get_nic_data_by_name(self, name):
    nics = self._conn.Msvm_SyntheticEthernetPortSettingData(ElementName=name)
    if nics:
        return nics[0]
    raise exceptions.HyperVvNicNotFound(vnic_name=name)