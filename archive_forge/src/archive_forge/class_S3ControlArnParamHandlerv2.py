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
class S3ControlArnParamHandlerv2(S3ControlArnParamHandler):
    """Updated version of S3ControlArnParamHandler for use when
    EndpointRulesetResolver is in use for endpoint resolution.

    This class is considered private and subject to abrupt breaking changes or
    removal without prior announcement. Please do not use it directly.
    """

    def __init__(self, arn_parser=None):
        self._arn_parser = arn_parser
        if arn_parser is None:
            self._arn_parser = ArnParser()

    def register(self, event_emitter):
        event_emitter.register('before-endpoint-resolution.s3-control', self.handle_arn)

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

    def _store_outpost_accesspoint(self, params, context, arn_details):
        self._override_account_id_param(params, arn_details)

    def _handle_bucket_param(self, params, model, context):
        arn_details = self._get_arn_details_from_param(params, 'Bucket')
        if arn_details is None:
            return
        self._raise_for_fips_pseudo_region(arn_details)
        self._raise_for_accelerate_endpoint(context)
        if self._is_outpost_bucket(arn_details):
            self._store_outpost_bucket(params, context, arn_details)
        else:
            error_msg = 'The Bucket parameter does not support the provided ARN'
            raise UnsupportedS3ControlArnError(arn=arn_details['original'], msg=error_msg)

    def _store_outpost_bucket(self, params, context, arn_details):
        self._override_account_id_param(params, arn_details)

    def _raise_for_fips_pseudo_region(self, arn_details):
        arn_region = arn_details['region']
        if arn_region.startswith('fips-') or arn_region.endswith('fips-'):
            raise UnsupportedS3ControlArnError(arn=arn_details['original'], msg='Invalid ARN, FIPS region not allowed in ARN.')

    def _raise_for_accelerate_endpoint(self, context):
        s3_config = context['client_config'].s3 or {}
        if s3_config.get('use_accelerate_endpoint'):
            raise UnsupportedS3ControlConfigurationError(msg='S3 control client does not support accelerate endpoints')