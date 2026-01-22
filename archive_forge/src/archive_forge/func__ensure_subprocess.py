import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _ensure_subprocess():
    global subprocess
    if not subprocess:
        try:
            from eventlet import patcher
            if patcher.already_patched:
                from eventlet.green import subprocess
            else:
                import subprocess
        except ImportError:
            import subprocess