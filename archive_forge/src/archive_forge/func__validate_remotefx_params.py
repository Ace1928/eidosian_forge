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
def _validate_remotefx_params(self, monitor_count, max_resolution, vram_bytes=None):
    super(VMUtils10, self)._validate_remotefx_params(monitor_count, max_resolution)
    if vram_bytes and vram_bytes not in self._remotefx_vram_vals:
        raise exceptions.HyperVRemoteFXException(_('Unsuported RemoteFX VRAM value: %(requested_value)s.The supported VRAM values are: %(supported_values)s') % {'requested_value': vram_bytes, 'supported_values': self._remotefx_vram_vals})