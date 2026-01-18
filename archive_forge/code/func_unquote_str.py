import os
from boto.vendored import six
from boto.vendored.six import BytesIO, StringIO
from boto.vendored.six.moves import filter, http_client, map, _thread, \
from boto.vendored.six.moves.queue import Queue
from boto.vendored.six.moves.urllib.parse import parse_qs, quote, unquote, \
from boto.vendored.six.moves.urllib.parse import unquote_plus
from boto.vendored.six.moves.urllib.request import urlopen
def unquote_str(value, encoding='utf-8'):
    byte_string = value.encode(encoding)
    return unquote_plus(byte_string).decode(encoding)