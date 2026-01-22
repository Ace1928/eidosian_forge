import ssl
class SingleNetstringFetcher:

    def __init__(self, incoming, maxlen=-1):
        self._incoming = incoming
        self._maxlen = maxlen
        self._len_known = False
        self._len = None
        self._done = False
        self._length_bytes = b''

    def done(self):
        return self._done

    def pending(self):
        return self._len is not None

    def read(self, nbytes=65536):
        if not self._len_known:
            while True:
                symbol = self._incoming.read(1)
                if not symbol:
                    raise WantRead()
                if symbol == COLON:
                    if self._len is None:
                        raise BadLength('No netstring length digits seen.')
                    self._len_known = True
                    break
                if not symbol.isdigit():
                    raise BadLength('Non-digit symbol in netstring length.')
                val = ord(symbol) - ZERO_ORD
                self._len = val if self._len is None else self._len * 10 + val
                if self._maxlen != -1 and self._len > self._maxlen:
                    raise TooLong('Netstring length is over limit.')
        if self._len:
            buf = self._incoming.read(min(nbytes, self._len))
            if not buf:
                raise WantRead()
            self._len -= len(buf)
            return buf
        else:
            if not self._done:
                symbol = self._incoming.read(1)
                if not symbol:
                    raise WantRead()
                if symbol == COMMA:
                    self._done = True
                else:
                    raise BadTerminator('Bad netstring terminator.')
            return b''