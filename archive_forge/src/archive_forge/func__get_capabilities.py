import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def _get_capabilities(self):
    typ, dat = self.capability()
    if dat == [None]:
        raise self.error('no CAPABILITY response from server')
    dat = str(dat[-1], self._encoding)
    dat = dat.upper()
    self.capabilities = tuple(dat.split())