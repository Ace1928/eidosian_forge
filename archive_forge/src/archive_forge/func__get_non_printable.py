from __future__ import absolute_import
import codecs
from ruamel.yaml.error import YAMLError, FileMark, StringMark, YAMLStreamError
from ruamel.yaml.compat import text_type, binary_type, PY3, UNICODE_SIZE
from ruamel.yaml.util import RegExp
@classmethod
def _get_non_printable(cls, data):
    try:
        return cls._get_non_printable_ascii(data)
    except UnicodeEncodeError:
        return cls._get_non_printable_regex(data)