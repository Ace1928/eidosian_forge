import re
from email import errors
from email._policybase import compat32
from collections import deque
from io import StringIO
def _parsegen(self):
    self._new_message()
    headers = []
    for line in self._input:
        if line is NeedMoreData:
            yield NeedMoreData
            continue
        if not headerRE.match(line):
            if not NLCRE.match(line):
                defect = errors.MissingHeaderBodySeparatorDefect()
                self.policy.handle_defect(self._cur, defect)
                self._input.unreadline(line)
            break
        headers.append(line)
    self._parse_headers(headers)
    if self._headersonly:
        lines = []
        while True:
            line = self._input.readline()
            if line is NeedMoreData:
                yield NeedMoreData
                continue
            if line == '':
                break
            lines.append(line)
        self._cur.set_payload(EMPTYSTRING.join(lines))
        return
    if self._cur.get_content_type() == 'message/delivery-status':
        while True:
            self._input.push_eof_matcher(NLCRE.match)
            for retval in self._parsegen():
                if retval is NeedMoreData:
                    yield NeedMoreData
                    continue
                break
            msg = self._pop_message()
            self._input.pop_eof_matcher()
            while True:
                line = self._input.readline()
                if line is NeedMoreData:
                    yield NeedMoreData
                    continue
                break
            while True:
                line = self._input.readline()
                if line is NeedMoreData:
                    yield NeedMoreData
                    continue
                break
            if line == '':
                break
            self._input.unreadline(line)
        return
    if self._cur.get_content_maintype() == 'message':
        for retval in self._parsegen():
            if retval is NeedMoreData:
                yield NeedMoreData
                continue
            break
        self._pop_message()
        return
    if self._cur.get_content_maintype() == 'multipart':
        boundary = self._cur.get_boundary()
        if boundary is None:
            defect = errors.NoBoundaryInMultipartDefect()
            self.policy.handle_defect(self._cur, defect)
            lines = []
            for line in self._input:
                if line is NeedMoreData:
                    yield NeedMoreData
                    continue
                lines.append(line)
            self._cur.set_payload(EMPTYSTRING.join(lines))
            return
        if str(self._cur.get('content-transfer-encoding', '8bit')).lower() not in ('7bit', '8bit', 'binary'):
            defect = errors.InvalidMultipartContentTransferEncodingDefect()
            self.policy.handle_defect(self._cur, defect)
        separator = '--' + boundary
        boundaryre = re.compile('(?P<sep>' + re.escape(separator) + ')(?P<end>--)?(?P<ws>[ \\t]*)(?P<linesep>\\r\\n|\\r|\\n)?$')
        capturing_preamble = True
        preamble = []
        linesep = False
        close_boundary_seen = False
        while True:
            line = self._input.readline()
            if line is NeedMoreData:
                yield NeedMoreData
                continue
            if line == '':
                break
            mo = boundaryre.match(line)
            if mo:
                if mo.group('end'):
                    close_boundary_seen = True
                    linesep = mo.group('linesep')
                    break
                if capturing_preamble:
                    if preamble:
                        lastline = preamble[-1]
                        eolmo = NLCRE_eol.search(lastline)
                        if eolmo:
                            preamble[-1] = lastline[:-len(eolmo.group(0))]
                        self._cur.preamble = EMPTYSTRING.join(preamble)
                    capturing_preamble = False
                    self._input.unreadline(line)
                    continue
                while True:
                    line = self._input.readline()
                    if line is NeedMoreData:
                        yield NeedMoreData
                        continue
                    mo = boundaryre.match(line)
                    if not mo:
                        self._input.unreadline(line)
                        break
                self._input.push_eof_matcher(boundaryre.match)
                for retval in self._parsegen():
                    if retval is NeedMoreData:
                        yield NeedMoreData
                        continue
                    break
                if self._last.get_content_maintype() == 'multipart':
                    epilogue = self._last.epilogue
                    if epilogue == '':
                        self._last.epilogue = None
                    elif epilogue is not None:
                        mo = NLCRE_eol.search(epilogue)
                        if mo:
                            end = len(mo.group(0))
                            self._last.epilogue = epilogue[:-end]
                else:
                    payload = self._last._payload
                    if isinstance(payload, str):
                        mo = NLCRE_eol.search(payload)
                        if mo:
                            payload = payload[:-len(mo.group(0))]
                            self._last._payload = payload
                self._input.pop_eof_matcher()
                self._pop_message()
                self._last = self._cur
            else:
                assert capturing_preamble
                preamble.append(line)
        if capturing_preamble:
            defect = errors.StartBoundaryNotFoundDefect()
            self.policy.handle_defect(self._cur, defect)
            self._cur.set_payload(EMPTYSTRING.join(preamble))
            epilogue = []
            for line in self._input:
                if line is NeedMoreData:
                    yield NeedMoreData
                    continue
            self._cur.epilogue = EMPTYSTRING.join(epilogue)
            return
        if not close_boundary_seen:
            defect = errors.CloseBoundaryNotFoundDefect()
            self.policy.handle_defect(self._cur, defect)
            return
        if linesep:
            epilogue = ['']
        else:
            epilogue = []
        for line in self._input:
            if line is NeedMoreData:
                yield NeedMoreData
                continue
            epilogue.append(line)
        if epilogue:
            firstline = epilogue[0]
            bolmo = NLCRE_bol.match(firstline)
            if bolmo:
                epilogue[0] = firstline[len(bolmo.group(0)):]
        self._cur.epilogue = EMPTYSTRING.join(epilogue)
        return
    lines = []
    for line in self._input:
        if line is NeedMoreData:
            yield NeedMoreData
            continue
        lines.append(line)
    self._cur.set_payload(EMPTYSTRING.join(lines))