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
def _timestamp_iso8601(self, value):
    if value.microsecond > 0:
        timestamp_format = ISO8601_MICRO
    else:
        timestamp_format = ISO8601
    return value.strftime(timestamp_format)