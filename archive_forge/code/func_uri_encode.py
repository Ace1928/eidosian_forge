import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
def uri_encode(self, value):
    """Perform percent-encoding on an input string.

        :type value: str
        :rytpe: str
        """
    if value is None:
        return None
    return percent_encode(value)