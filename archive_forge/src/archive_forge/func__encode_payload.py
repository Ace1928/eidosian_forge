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
def _encode_payload(self, body):
    if isinstance(body, str):
        return body.encode(self.DEFAULT_ENCODING)
    return body