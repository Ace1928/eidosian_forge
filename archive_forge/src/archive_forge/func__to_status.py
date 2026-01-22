import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_status(self, element):
    if element is None:
        return NttCisStatus()
    s = NttCisStatus(action=findtext(element, 'action', TYPES_URN), request_time=findtext(element, 'requestTime', TYPES_URN), user_name=findtext(element, 'userName', TYPES_URN), number_of_steps=findtext(element, 'numberOfSteps', TYPES_URN), step_name=findtext(element, 'step/name', TYPES_URN), step_number=findtext(element, 'step_number', TYPES_URN), step_percent_complete=findtext(element, 'step/percentComplete', TYPES_URN), failure_reason=findtext(element, 'failureReason', TYPES_URN))
    return s