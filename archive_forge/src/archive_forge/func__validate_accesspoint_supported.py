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
def _validate_accesspoint_supported(self, request):
    if self._use_accelerate_endpoint:
        raise UnsupportedS3AccesspointConfigurationError(msg='Client does not support s3 accelerate configuration when an access-point ARN is specified.')
    request_partition = request.context['s3_accesspoint']['partition']
    if request_partition != self._partition:
        raise UnsupportedS3AccesspointConfigurationError(msg='Client is configured for "%s" partition, but access-point ARN provided is for "%s" partition. The client and  access-point partition must be the same.' % (self._partition, request_partition))
    s3_service = request.context['s3_accesspoint'].get('service')
    if s3_service == 's3-object-lambda' and self._s3_config.get('use_dualstack_endpoint'):
        raise UnsupportedS3AccesspointConfigurationError(msg='Client does not support s3 dualstack configuration when an S3 Object Lambda access point ARN is specified.')
    outpost_name = request.context['s3_accesspoint'].get('outpost_name')
    if outpost_name and self._s3_config.get('use_dualstack_endpoint'):
        raise UnsupportedS3AccesspointConfigurationError(msg='Client does not support s3 dualstack configuration when an outpost ARN is specified.')
    self._validate_mrap_s3_config(request)