import logging
import operator
import os
import re
import sys
import weakref
import ovs.db.data
import ovs.db.parser
import ovs.db.schema
import ovs.db.types
import ovs.poller
import ovs.json
from ovs import jsonrpc
from ovs import ovsuuid
from ovs import stream
from ovs.db import idl
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.ovs import vswitch_idl
from os_ken.lib.stringify import StringifyMixin
@staticmethod
def _score_partial_match(name, s):
    _MAX_SCORE = 4294967295
    assert len(name) < _MAX_SCORE
    s = s[:_MAX_SCORE - 1]
    if name == s:
        return _MAX_SCORE
    name = name.lower().replace('-', '_')
    s = s.lower().replace('-', '_')
    if s.startswith(name):
        return _MAX_SCORE - 1
    if name.startswith(s):
        return len(s)
    return 0