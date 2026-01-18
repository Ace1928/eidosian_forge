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
def this_boot(self, bootid=None):
    """Add match for _BOOT_ID for current boot or the specified boot ID.

        If specified, bootid should be either a UUID or a 32 digit hex number.

        Equivalent to add_match(_BOOT_ID='bootid').
        """
    if bootid is None:
        bootid = _id128.get_boot().hex
    else:
        bootid = getattr(bootid, 'hex', bootid)
    self.add_match(_BOOT_ID=bootid)