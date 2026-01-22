import stat
import time
from paramiko.common import x80000000, o700, o70, xffffffff
@classmethod
def _from_msg(cls, msg, filename=None, longname=None):
    attr = cls()
    attr._unpack(msg)
    if filename is not None:
        attr.filename = filename
    if longname is not None:
        attr.longname = longname
    return attr