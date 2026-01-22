import base64
import binascii
import datetime
import email.message
import functools
import hashlib
import io
import logging
import os
import random
import re
import socket
import time
import warnings
import weakref
from datetime import datetime as _DatetimeClass
from ipaddress import ip_address
from pathlib import Path
from urllib.request import getproxies, proxy_bypass
import dateutil.parser
from dateutil.tz import tzutc
from urllib3.exceptions import LocationParseError
import botocore
import botocore.awsrequest
import botocore.httpsession
from botocore.compat import HEX_PAT  # noqa: F401
from botocore.compat import IPV4_PAT  # noqa: F401
from botocore.compat import IPV6_ADDRZ_PAT  # noqa: F401
from botocore.compat import IPV6_PAT  # noqa: F401
from botocore.compat import LS32_PAT  # noqa: F401
from botocore.compat import UNRESERVED_PAT  # noqa: F401
from botocore.compat import ZONE_ID_PAT  # noqa: F401
from botocore.compat import (
from botocore.exceptions import (
def _handle_name_param(self, params, model, context):
    if model.name == 'CreateAccessPoint':
        return
    arn_details = self._get_arn_details_from_param(params, 'Name')
    if arn_details is None:
        return
    self._raise_for_fips_pseudo_region(arn_details)
    self._raise_for_accelerate_endpoint(context)
    if self._is_outpost_accesspoint(arn_details):
        self._store_outpost_accesspoint(params, context, arn_details)
    else:
        error_msg = 'The Name parameter does not support the provided ARN'
        raise UnsupportedS3ControlArnError(arn=arn_details['original'], msg=error_msg)