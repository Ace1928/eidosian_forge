from __future__ import division
import sys as _sys
import datetime as _datetime
import uuid as _uuid
import traceback as _traceback
import os as _os
import logging as _logging
from syslog import (LOG_EMERG, LOG_ALERT, LOG_CRIT, LOG_ERR,
from ._journal import __version__, sendv, stream_fd
from ._reader import (_Reader, NOP, APPEND, INVALIDATE,
from . import id128 as _id128
def this_machine(self, machineid=None):
    """Add match for _MACHINE_ID equal to the ID of this machine.

        If specified, machineid should be either a UUID or a 32 digit hex
        number.

        Equivalent to add_match(_MACHINE_ID='machineid').
        """
    if machineid is None:
        machineid = _id128.get_machine().hex
    else:
        machineid = getattr(machineid, 'hex', machineid)
    self.add_match(_MACHINE_ID=machineid)