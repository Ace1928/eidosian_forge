from .core import encode, decode, alabel, ulabel, IDNAError
import codecs
import re
from typing import Any, Tuple, Optional
class IncrementalEncoder(codecs.BufferedIncrementalEncoder):

    def _buffer_encode(self, data: str, errors: str, final: bool) -> Tuple[bytes, int]:
        if errors != 'strict':
            raise IDNAError('Unsupported error handling "{}"'.format(errors))
        if not data:
            return (b'', 0)
        labels = _unicode_dots_re.split(data)
        trailing_dot = b''
        if labels:
            if not labels[-1]:
                trailing_dot = b'.'
                del labels[-1]
            elif not final:
                del labels[-1]
                if labels:
                    trailing_dot = b'.'
        result = []
        size = 0
        for label in labels:
            result.append(alabel(label))
            if size:
                size += 1
            size += len(label)
        result_bytes = b'.'.join(result) + trailing_dot
        size += len(trailing_dot)
        return (result_bytes, size)