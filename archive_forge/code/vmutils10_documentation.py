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
Removes all the PCI devices from the given VM.

        :param vm_name: the name of the VM from which all the PCI devices will
            be detached from.
        