import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def _get_response(self):
    resp = self._get_line()
    if self._match(self.tagre, resp):
        tag = self.mo.group('tag')
        if not tag in self.tagged_commands:
            raise self.abort('unexpected tagged response: %r' % resp)
        typ = self.mo.group('type')
        typ = str(typ, self._encoding)
        dat = self.mo.group('data')
        self.tagged_commands[tag] = (typ, [dat])
    else:
        dat2 = None
        if not self._match(Untagged_response, resp):
            if self._match(self.Untagged_status, resp):
                dat2 = self.mo.group('data2')
        if self.mo is None:
            if self._match(Continuation, resp):
                self.continuation_response = self.mo.group('data')
                return None
            raise self.abort('unexpected response: %r' % resp)
        typ = self.mo.group('type')
        typ = str(typ, self._encoding)
        dat = self.mo.group('data')
        if dat is None:
            dat = b''
        if dat2:
            dat = dat + b' ' + dat2
        while self._match(self.Literal, dat):
            size = int(self.mo.group('size'))
            if __debug__:
                if self.debug >= 4:
                    self._mesg('read literal size %s' % size)
            data = self.read(size)
            self._append_untagged(typ, (dat, data))
            dat = self._get_line()
        self._append_untagged(typ, dat)
    if typ in ('OK', 'NO', 'BAD') and self._match(Response_code, dat):
        typ = self.mo.group('type')
        typ = str(typ, self._encoding)
        self._append_untagged(typ, self.mo.group('data'))
    if __debug__:
        if self.debug >= 1 and typ in ('NO', 'BAD', 'BYE'):
            self._mesg('%s response: %r' % (typ, dat))
    return resp