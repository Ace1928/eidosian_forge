import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
def _decode_host(host):
    """Decode a host from ASCII-encodable text to IDNA-decoded text. If
    the host text is not ASCII, it is returned unchanged, as it is
    presumed that it is already IDNA-decoded.

    Some technical details: _decode_host is built on top of the "idna"
    package, which has some quirks:

    Capital letters are not valid IDNA2008. The idna package will
    raise an exception like this on capital letters:

    > idna.core.InvalidCodepoint: Codepoint U+004B at position 1 ... not allowed

    However, if a segment of a host (i.e., something in
    url.host.split('.')) is already ASCII, idna doesn't perform its
    usual checks. In fact, for capital letters it automatically
    lowercases them.

    This check and some other functionality can be bypassed by passing
    uts46=True to idna.encode/decode. This allows a more permissive and
    convenient interface. So far it seems like the balanced approach.

    Example output (from idna==2.6):

    >> idna.encode(u'mahmöud.io')
    'xn--mahmud-zxa.io'
    >> idna.encode(u'Mahmöud.io')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/home/mahmoud/virtualenvs/hyperlink/local/lib/python2.7/site-packages/idna/core.py", line 355, in encode
        result.append(alabel(label))
      File "/home/mahmoud/virtualenvs/hyperlink/local/lib/python2.7/site-packages/idna/core.py", line 276, in alabel
        check_label(label)
      File "/home/mahmoud/virtualenvs/hyperlink/local/lib/python2.7/site-packages/idna/core.py", line 253, in check_label
        raise InvalidCodepoint('Codepoint {0} at position {1} of {2} not allowed'.format(_unot(cp_value), pos+1, repr(label)))
    idna.core.InvalidCodepoint: Codepoint U+004D at position 1 of u'Mahmöud' not allowed
    >> idna.encode(u'Mahmoud.io')
    'Mahmoud.io'

    # Similar behavior for decodes below
    >> idna.decode(u'Mahmoud.io')
    u'mahmoud.io
    >> idna.decode(u'Méhmoud.io', uts46=True)
    u'méhmoud.io'
    """
    if not host:
        return u''
    try:
        host_bytes = host.encode('ascii')
    except UnicodeEncodeError:
        host_text = host
    else:
        try:
            host_text = idna_decode(host_bytes, uts46=True)
        except ValueError:
            host_text = host
    return host_text