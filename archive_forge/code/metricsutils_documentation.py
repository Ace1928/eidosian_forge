from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils

Utility class for metrics related operations.
Based on the "root/virtualization/v2" namespace available starting with
Hyper-V Server / Windows Server 2012.
