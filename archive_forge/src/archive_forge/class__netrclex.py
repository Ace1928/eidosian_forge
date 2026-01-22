import os, shlex, stat
class _netrclex:

    def __init__(self, fp):
        self.lineno = 1
        self.instream = fp
        self.whitespace = '\n\t\r '
        self.pushback = []

    def _read_char(self):
        ch = self.instream.read(1)
        if ch == '\n':
            self.lineno += 1
        return ch

    def get_token(self):
        if self.pushback:
            return self.pushback.pop(0)
        token = ''
        fiter = iter(self._read_char, '')
        for ch in fiter:
            if ch in self.whitespace:
                continue
            if ch == '"':
                for ch in fiter:
                    if ch == '"':
                        return token
                    elif ch == '\\':
                        ch = self._read_char()
                    token += ch
            else:
                if ch == '\\':
                    ch = self._read_char()
                token += ch
                for ch in fiter:
                    if ch in self.whitespace:
                        return token
                    elif ch == '\\':
                        ch = self._read_char()
                    token += ch
        return token

    def push_token(self, token):
        self.pushback.append(token)