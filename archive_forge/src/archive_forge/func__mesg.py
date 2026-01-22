import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def _mesg(self, s, secs=None):
    if secs is None:
        secs = time.time()
    tm = time.strftime('%M:%S', time.localtime(secs))
    sys.stderr.write('  %s.%02d %s\n' % (tm, secs * 100 % 100, s))
    sys.stderr.flush()