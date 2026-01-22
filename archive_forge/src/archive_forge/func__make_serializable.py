import json
import os
import sys
def _make_serializable(obj):
    """Converts objects to serializable form."""
    if isinstance(obj, JSONObject):
        return obj.to_dict()
    else:
        return obj