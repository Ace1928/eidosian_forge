from __future__ import absolute_import
import codecs
from ruamel.yaml.error import YAMLError, FileMark, StringMark, YAMLStreamError
from ruamel.yaml.compat import text_type, binary_type, PY3, UNICODE_SIZE
from ruamel.yaml.util import RegExp
@classmethod
def _get_non_printable_ascii(cls, data):
    ascii_bytes = data.encode('ascii')
    non_printables = ascii_bytes.translate(None, cls._printable_ascii)
    if not non_printables:
        return None
    non_printable = non_printables[:1]
    return (ascii_bytes.index(non_printable), non_printable.decode('ascii'))