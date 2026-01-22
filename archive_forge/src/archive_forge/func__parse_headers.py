import re
from email import errors
from email._policybase import compat32
from collections import deque
from io import StringIO
def _parse_headers(self, lines):
    lastheader = ''
    lastvalue = []
    for lineno, line in enumerate(lines):
        if line[0] in ' \t':
            if not lastheader:
                defect = errors.FirstHeaderLineIsContinuationDefect(line)
                self.policy.handle_defect(self._cur, defect)
                continue
            lastvalue.append(line)
            continue
        if lastheader:
            self._cur.set_raw(*self.policy.header_source_parse(lastvalue))
            lastheader, lastvalue = ('', [])
        if line.startswith('From '):
            if lineno == 0:
                mo = NLCRE_eol.search(line)
                if mo:
                    line = line[:-len(mo.group(0))]
                self._cur.set_unixfrom(line)
                continue
            elif lineno == len(lines) - 1:
                self._input.unreadline(line)
                return
            else:
                defect = errors.MisplacedEnvelopeHeaderDefect(line)
                self._cur.defects.append(defect)
                continue
        i = line.find(':')
        if i == 0:
            defect = errors.InvalidHeaderDefect('Missing header name.')
            self._cur.defects.append(defect)
            continue
        assert i > 0, '_parse_headers fed line with no : and no leading WS'
        lastheader = line[:i]
        lastvalue = [line]
    if lastheader:
        self._cur.set_raw(*self.policy.header_source_parse(lastvalue))