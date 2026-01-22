from __future__ import absolute_import, division, print_function
import codecs
import datetime
import json
from ansible.module_utils.six.moves.collections_abc import Set
from ansible.module_utils.six import (
def _json_encode_fallback(obj):
    if isinstance(obj, Set):
        return list(obj)
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    raise TypeError('Cannot json serialize %s' % to_native(obj))