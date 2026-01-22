import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
@staticmethod
def _get_node_state(state, started, action):
    try:
        return NODE_STATE_MAP[state, started, action]
    except KeyError:
        if started == 'true':
            return NodeState.RUNNING
        else:
            return NodeState.TERMINATED