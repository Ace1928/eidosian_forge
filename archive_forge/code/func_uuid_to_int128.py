import base64
import hashlib
import hmac
import json
import os
import uuid
from oslo_utils import secretutils
from oslo_utils import uuidutils
def uuid_to_int128(span_uuid):
    """Convert from uuid4 to 128 bit id for OpenTracing"""
    if isinstance(span_uuid, int):
        return span_uuid
    try:
        span_int = uuid.UUID(span_uuid).int
    except ValueError:
        span_int = uuid_to_int128(uuidutils.generate_uuid())
    return span_int