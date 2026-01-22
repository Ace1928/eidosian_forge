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
def _set_secure_boot(self, vs_data, msft_ca_required):
    vs_data.SecureBootEnabled = True
    if msft_ca_required:
        uefi_data = self._conn.Msvm_VirtualSystemSettingData(ElementName=self._UEFI_CERTIFICATE_AUTH)[0]
        vs_data.SecureBootTemplateId = uefi_data.SecureBootTemplateId