from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future.builtins import int, range, str, super, list
import re
from collections import namedtuple, OrderedDict
from future.backports.urllib.parse import (unquote, unquote_to_bytes)
from future.backports.email import _encoded_words as _ew
from future.backports.email import errors
from future.backports.email import utils
from the input.  Thus a parser method consumes the next syntactic construct
def append_if_fits(self, token, stoken=None):
    if stoken is None:
        stoken = str(token)
    l = len(stoken)
    if self.stickyspace is not None:
        stickyspace_len = len(self.stickyspace)
        if self.lastlen + stickyspace_len + l <= self.maxlen:
            self.current.append(self.stickyspace)
            self.lastlen += stickyspace_len
            self.current.append(stoken)
            self.lastlen += l
            self.stickyspace = None
            self.firstline = False
            return True
        if token.has_fws:
            ws = token.pop_leading_fws()
            if ws is not None:
                self.stickyspace += str(ws)
                stickyspace_len += len(ws)
            token._fold(self)
            return True
        if stickyspace_len and l + 1 <= self.maxlen:
            margin = self.maxlen - l
            if 0 < margin < stickyspace_len:
                trim = stickyspace_len - margin
                self.current.append(self.stickyspace[:trim])
                self.stickyspace = self.stickyspace[trim:]
                stickyspace_len = trim
            self.newline()
            self.current.append(self.stickyspace)
            self.current.append(stoken)
            self.lastlen = l + stickyspace_len
            self.stickyspace = None
            self.firstline = False
            return True
        if not self.firstline:
            self.newline()
        self.current.append(self.stickyspace)
        self.current.append(stoken)
        self.stickyspace = None
        self.firstline = False
        return True
    if self.lastlen + l <= self.maxlen:
        self.current.append(stoken)
        self.lastlen += l
        return True
    if l < self.maxlen:
        self.newline()
        self.current.append(stoken)
        self.lastlen = l
        return True
    return False