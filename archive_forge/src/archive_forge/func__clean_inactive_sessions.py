from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
@staticmethod
def _clean_inactive_sessions():
    """Removes sessions which are inactive more than 20 min"""
    session_cache = sessionDict
    logger.debug('cleaning inactive sessions in pid %d num elem %d', os.getpid(), len(session_cache))
    keys_to_delete = []
    for key, session in list(session_cache.items()):
        tdiff = avi_timedelta(datetime.utcnow() - session['last_used'])
        if tdiff < ApiSession.SESSION_CACHE_EXPIRY:
            continue
        keys_to_delete.append(key)
    for key in keys_to_delete:
        del session_cache[key]
        logger.debug('Removed session for : %s', key)