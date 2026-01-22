from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
def _get_streaming_body(self, shape):
    """Returns the streaming member's shape if any; or None otherwise."""
    if shape is None:
        return None
    payload = shape.serialization.get('payload')
    if payload is not None:
        payload_shape = shape.members[payload]
        if payload_shape.type_name == 'blob':
            return payload_shape
    return None