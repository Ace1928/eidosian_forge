from __future__ import annotations
import sys
import eventlet
def _green_MySQLdb():
    try:
        from eventlet.green import MySQLdb
        return [('MySQLdb', MySQLdb)]
    except ImportError:
        return []