import base64
import binascii
import collections
import html
from django.conf import settings
from django.core.exceptions import (
from django.core.files.uploadhandler import SkipFile, StopFutureHandlers, StopUpload
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.http import parse_header_parameters
from django.utils.regex_helper import _lazy_re_compile
def _update_unget_history(self, num_bytes):
    """
        Update the unget history as a sanity check to see if we've pushed
        back the same number of bytes in one chunk. If we keep ungetting the
        same number of bytes many times (here, 50), we're mostly likely in an
        infinite loop of some sort. This is usually caused by a
        maliciously-malformed MIME request.
        """
    self._unget_history = [num_bytes] + self._unget_history[:49]
    number_equal = len([current_number for current_number in self._unget_history if current_number == num_bytes])
    if number_equal > 40:
        raise SuspiciousMultipartForm("The multipart parser got stuck, which shouldn't happen with normal uploaded files. Check for malicious upload activity; if there is none, report this to the Django developers.")