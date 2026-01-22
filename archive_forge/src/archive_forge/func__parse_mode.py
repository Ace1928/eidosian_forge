import array
from typing import (
def _parse_mode(self, mode: object) -> BitParserState:
    if mode == 'p':
        self._do_pass()
        self._flush_line()
        return self.MODE
    elif mode == 'h':
        self._n1 = 0
        self._accept = self._parse_horiz1
        if self._color:
            return self.WHITE
        else:
            return self.BLACK
    elif mode == 'u':
        self._accept = self._parse_uncompressed
        return self.UNCOMPRESSED
    elif mode == 'e':
        raise self.EOFB
    elif isinstance(mode, int):
        self._do_vertical(mode)
        self._flush_line()
        return self.MODE
    else:
        raise self.InvalidData(mode)