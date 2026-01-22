import hashlib
import re
import uuid
import warnings
from oslo_config import cfg
import six
def _check_valid_uuid(value):
    """Checks a value for one or multiple valid uuids joined together."""
    if not value:
        raise ValueError
    value = re.sub('[{}-]|urn:uuid:', '', value)
    for val in [value[i:i + 32] for i in range(0, len(value), 32)]:
        uuid.UUID(val)