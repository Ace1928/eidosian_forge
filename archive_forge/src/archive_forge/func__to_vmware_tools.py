import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_vmware_tools(self, element):
    status = None
    if hasattr(element, 'runningStatus'):
        status = element.get('runningStatus')
    version_status = None
    if hasattr(element, 'version_status'):
        version_status = element.get('version_status')
    api_version = None
    if hasattr(element, 'apiVersion'):
        api_version = element.get('apiVersion')
    return NttCisServerVMWareTools(status=status, version_status=version_status, api_version=api_version)