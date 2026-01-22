import re
import binascii
import email.quoprimime
import email.base64mime
from email.errors import HeaderParseError
from email import charset as _charset
class _ValueFormatter:

    def __init__(self, headerlen, maxlen, continuation_ws, splitchars):
        self._maxlen = maxlen
        self._continuation_ws = continuation_ws
        self._continuation_ws_len = len(continuation_ws)
        self._splitchars = splitchars
        self._lines = []
        self._current_line = _Accumulator(headerlen)

    def _str(self, linesep):
        self.newline()
        return linesep.join(self._lines)

    def __str__(self):
        return self._str(NL)

    def newline(self):
        end_of_line = self._current_line.pop()
        if end_of_line != (' ', ''):
            self._current_line.push(*end_of_line)
        if len(self._current_line) > 0:
            if self._current_line.is_onlyws() and self._lines:
                self._lines[-1] += str(self._current_line)
            else:
                self._lines.append(str(self._current_line))
        self._current_line.reset()

    def add_transition(self):
        self._current_line.push(' ', '')

    def feed(self, fws, string, charset):
        if charset.header_encoding is None:
            self._ascii_split(fws, string, self._splitchars)
            return
        encoded_lines = charset.header_encode_lines(string, self._maxlengths())
        try:
            first_line = encoded_lines.pop(0)
        except IndexError:
            return
        if first_line is not None:
            self._append_chunk(fws, first_line)
        try:
            last_line = encoded_lines.pop()
        except IndexError:
            return
        self.newline()
        self._current_line.push(self._continuation_ws, last_line)
        for line in encoded_lines:
            self._lines.append(self._continuation_ws + line)

    def _maxlengths(self):
        yield (self._maxlen - len(self._current_line))
        while True:
            yield (self._maxlen - self._continuation_ws_len)

    def _ascii_split(self, fws, string, splitchars):
        parts = re.split('([' + FWS + ']+)', fws + string)
        if parts[0]:
            parts[:0] = ['']
        else:
            parts.pop(0)
        for fws, part in zip(*[iter(parts)] * 2):
            self._append_chunk(fws, part)

    def _append_chunk(self, fws, string):
        self._current_line.push(fws, string)
        if len(self._current_line) > self._maxlen:
            for ch in self._splitchars:
                for i in range(self._current_line.part_count() - 1, 0, -1):
                    if ch.isspace():
                        fws = self._current_line[i][0]
                        if fws and fws[0] == ch:
                            break
                    prevpart = self._current_line[i - 1][1]
                    if prevpart and prevpart[-1] == ch:
                        break
                else:
                    continue
                break
            else:
                fws, part = self._current_line.pop()
                if self._current_line._initial_size > 0:
                    self.newline()
                    if not fws:
                        fws = ' '
                self._current_line.push(fws, part)
                return
            remainder = self._current_line.pop_from(i)
            self._lines.append(str(self._current_line))
            self._current_line.reset(remainder)