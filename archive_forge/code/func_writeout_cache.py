import atexit
import datetime
import re
import sqlite3
import threading
from pathlib import Path
from decorator import decorator
from traitlets import (
from traitlets.config.configurable import LoggingConfigurable
from IPython.paths import locate_profile
from IPython.utils.decorators import undoc
@only_when_enabled
def writeout_cache(self, conn=None):
    """Write any entries in the cache to the database."""
    if conn is None:
        conn = self.db
    with self.db_input_cache_lock:
        try:
            self._writeout_input_cache(conn)
        except sqlite3.IntegrityError:
            self.new_session(conn)
            print('ERROR! Session/line number was not unique in', 'database. History logging moved to new session', self.session_number)
            try:
                self._writeout_input_cache(conn)
            except sqlite3.IntegrityError:
                pass
        finally:
            self.db_input_cache = []
    with self.db_output_cache_lock:
        try:
            self._writeout_output_cache(conn)
        except sqlite3.IntegrityError:
            print('!! Session/line number for output was not unique', 'in database. Output will not be stored.')
        finally:
            self.db_output_cache = []