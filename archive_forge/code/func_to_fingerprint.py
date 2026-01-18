from __future__ import absolute_import
import os
from .. import Utils
def to_fingerprint(item):
    """
            Recursively turn item into a string, turning dicts into lists with
            deterministic ordering.
            """
    if isinstance(item, dict):
        item = sorted([(repr(key), to_fingerprint(value)) for key, value in item.items()])
    return repr(item)