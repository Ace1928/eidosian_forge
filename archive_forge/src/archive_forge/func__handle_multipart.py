import re
import sys
import time
import random
from copy import deepcopy
from io import StringIO, BytesIO
from email.utils import _has_surrogates
def _handle_multipart(self, msg):
    msgtexts = []
    subparts = msg.get_payload()
    if subparts is None:
        subparts = []
    elif isinstance(subparts, str):
        self.write(subparts)
        return
    elif not isinstance(subparts, list):
        subparts = [subparts]
    for part in subparts:
        s = self._new_buffer()
        g = self.clone(s)
        g.flatten(part, unixfrom=False, linesep=self._NL)
        msgtexts.append(s.getvalue())
    boundary = msg.get_boundary()
    if not boundary:
        alltext = self._encoded_NL.join(msgtexts)
        boundary = self._make_boundary(alltext)
        msg.set_boundary(boundary)
    if msg.preamble is not None:
        if self._mangle_from_:
            preamble = fcre.sub('>From ', msg.preamble)
        else:
            preamble = msg.preamble
        self._write_lines(preamble)
        self.write(self._NL)
    self.write('--' + boundary + self._NL)
    if msgtexts:
        self._fp.write(msgtexts.pop(0))
    for body_part in msgtexts:
        self.write(self._NL + '--' + boundary + self._NL)
        self._fp.write(body_part)
    self.write(self._NL + '--' + boundary + '--' + self._NL)
    if msg.epilogue is not None:
        if self._mangle_from_:
            epilogue = fcre.sub('>From ', msg.epilogue)
        else:
            epilogue = msg.epilogue
        self._write_lines(epilogue)