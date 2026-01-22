import datetime
import errno
import os
import os.path
import time
def _ngettext(n, singular, plural):
    return '%d %s ago' % (n, singular if n == 1 else plural)