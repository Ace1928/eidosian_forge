import sys
import datetime
import os
import time
import threading
import binascii
import pickle
import zc.lockfile
import cherrypy
from cherrypy.lib import httputil
from cherrypy.lib import locking
from cherrypy.lib import is_iterator
def _regenerate(self):
    if self.id is not None:
        if self.debug:
            cherrypy.log('Deleting the existing session %r before regeneration.' % self.id, 'TOOLS.SESSIONS')
        self.delete()
    old_session_was_locked = self.locked
    if old_session_was_locked:
        self.release_lock()
        if self.debug:
            cherrypy.log('Old lock released.', 'TOOLS.SESSIONS')
    self.id = None
    while self.id is None:
        self.id = self.generate_id()
        if self._exists():
            self.id = None
    if self.debug:
        cherrypy.log('Set id to generated %s.' % self.id, 'TOOLS.SESSIONS')
    if old_session_was_locked:
        self.acquire_lock()
        if self.debug:
            cherrypy.log('Regenerated lock acquired.', 'TOOLS.SESSIONS')