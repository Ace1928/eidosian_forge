import base64
import calendar
import datetime
import json
import re
from xml.etree import ElementTree
from botocore import validate
from botocore.compat import formatdate
from botocore.exceptions import ParamValidationError
from botocore.utils import (
def _has_streaming_payload(self, payload, shape_members):
    """Determine if payload is streaming (a blob or string)."""
    return payload is not None and shape_members[payload].type_name in ('blob', 'string')