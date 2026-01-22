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
def _validate_mrap_s3_config(self, request):
    if not is_global_accesspoint(request.context):
        return
    if self._s3_config.get('s3_disable_multiregion_access_points'):
        raise UnsupportedS3AccesspointConfigurationError(msg='Invalid configuration, Multi-Region Access Point ARNs are disabled.')
    elif self._s3_config.get('use_dualstack_endpoint'):
        raise UnsupportedS3AccesspointConfigurationError(msg='Client does not support s3 dualstack configuration when a Multi-Region Access Point ARN is specified.')