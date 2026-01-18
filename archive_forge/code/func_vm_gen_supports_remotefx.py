import re
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from oslo_utils import units
def vm_gen_supports_remotefx(self, vm_gen):
    """RemoteFX is supported on both generation 1 and 2 virtual

        machines for Windows 10 / Windows Server 2016.

        :returns: True
        """
    return True