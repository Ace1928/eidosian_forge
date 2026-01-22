from __future__ import absolute_import
import re
from collections import namedtuple
from ..exceptions import LocationParseError
from ..packages import six
def _idna_encode(name):
    if name and any([ord(x) > 128 for x in name]):
        try:
            import idna
        except ImportError:
            six.raise_from(LocationParseError("Unable to parse URL without the 'idna' module"), None)
        try:
            return idna.encode(name.lower(), strict=True, std3_rules=True)
        except idna.IDNAError:
            six.raise_from(LocationParseError(u"Name '%s' is not a valid IDNA label" % name), None)
    return name.lower().encode('ascii')