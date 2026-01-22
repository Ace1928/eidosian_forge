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
class S3ExpressIdentityResolver:

    def __init__(self, client, credential_cls, cache=None):
        self._client = weakref.proxy(client)
        if cache is None:
            cache = S3ExpressIdentityCache(self._client, credential_cls)
        self._cache = cache

    def register(self, event_emitter=None):
        logger.debug('Registering S3Express Identity Resolver')
        emitter = event_emitter or self._client.meta.events
        emitter.register('before-parameter-build.s3', self.inject_signing_cache_key)
        emitter.register('before-call.s3', self.apply_signing_cache_key)
        emitter.register('before-sign.s3', self.resolve_s3express_identity)

    def inject_signing_cache_key(self, params, context, **kwargs):
        if 'Bucket' in params:
            context['S3Express'] = {'bucket_name': params['Bucket']}

    def apply_signing_cache_key(self, params, context, **kwargs):
        endpoint_properties = context.get('endpoint_properties', {})
        backend = endpoint_properties.get('backend', None)
        bucket_name = context.get('S3Express', {}).get('bucket_name')
        if backend == 'S3Express' and bucket_name is not None:
            context.setdefault('signing', {})
            context['signing']['cache_key'] = bucket_name

    def resolve_s3express_identity(self, request, signing_name, region_name, signature_version, request_signer, operation_name, **kwargs):
        signing_context = request.context.get('signing', {})
        signing_name = signing_context.get('signing_name')
        if signing_name == 's3express' and signature_version.startswith('v4-s3express'):
            signing_context['identity_cache'] = self._cache
            if 'cache_key' not in signing_context:
                signing_context['cache_key'] = request.context.get('s3_redirect', {}).get('params', {}).get('Bucket')