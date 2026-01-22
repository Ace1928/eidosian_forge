import urllib
from oslo_log import log as logging
from oslo_utils import timeutils
from glance.common import exception
from glance.i18n import _, _LE
def _call_callback(self, chunk, is_last=False):
    self._total_bytes += len(chunk)
    self._chunk_bytes += len(chunk)
    if not self._chunk_bytes:
        return
    if is_last or self.callback_due:
        self._callback(self._chunk_bytes, self._total_bytes)
        self._chunk_bytes = 0