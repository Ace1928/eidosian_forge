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
def _get_accesspoint_netloc(self, request_context, region_name):
    s3_accesspoint = request_context['s3_accesspoint']
    accesspoint_netloc_components = ['{}-{}'.format(s3_accesspoint['name'], s3_accesspoint['account'])]
    outpost_name = s3_accesspoint.get('outpost_name')
    if self._endpoint_url:
        if outpost_name:
            accesspoint_netloc_components.append(outpost_name)
        endpoint_url_netloc = urlsplit(self._endpoint_url).netloc
        accesspoint_netloc_components.append(endpoint_url_netloc)
    else:
        if outpost_name:
            outpost_host = [outpost_name, 's3-outposts']
            accesspoint_netloc_components.extend(outpost_host)
        elif s3_accesspoint['service'] == 's3-object-lambda':
            component = self._inject_fips_if_needed('s3-object-lambda', request_context)
            accesspoint_netloc_components.append(component)
        else:
            component = self._inject_fips_if_needed('s3-accesspoint', request_context)
            accesspoint_netloc_components.append(component)
        if self._s3_config.get('use_dualstack_endpoint'):
            accesspoint_netloc_components.append('dualstack')
        accesspoint_netloc_components.extend([region_name, self._get_dns_suffix(region_name)])
    return '.'.join(accesspoint_netloc_components)