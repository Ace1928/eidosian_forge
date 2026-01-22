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
class S3ControlEndpointSetter:
    _DEFAULT_PARTITION = 'aws'
    _DEFAULT_DNS_SUFFIX = 'amazonaws.com'
    _HOST_LABEL_REGEX = re.compile('^[a-zA-Z0-9\\-]{1,63}$')

    def __init__(self, endpoint_resolver, region=None, s3_config=None, endpoint_url=None, partition=None, use_fips_endpoint=False):
        self._endpoint_resolver = endpoint_resolver
        self._region = region
        self._s3_config = s3_config
        self._use_fips_endpoint = use_fips_endpoint
        if s3_config is None:
            self._s3_config = {}
        self._endpoint_url = endpoint_url
        self._partition = partition
        if partition is None:
            self._partition = self._DEFAULT_PARTITION

    def register(self, event_emitter):
        event_emitter.register('before-sign.s3-control', self.set_endpoint)

    def set_endpoint(self, request, **kwargs):
        if self._use_endpoint_from_arn_details(request):
            self._validate_endpoint_from_arn_details_supported(request)
            region_name = self._resolve_region_from_arn_details(request)
            self._resolve_signing_name_from_arn_details(request)
            self._resolve_endpoint_from_arn_details(request, region_name)
            self._add_headers_from_arn_details(request)
        elif self._use_endpoint_from_outpost_id(request):
            self._validate_outpost_redirection_valid(request)
            self._override_signing_name(request, 's3-outposts')
            new_netloc = self._construct_outpost_endpoint(self._region)
            self._update_request_netloc(request, new_netloc)

    def _use_endpoint_from_arn_details(self, request):
        return 'arn_details' in request.context

    def _use_endpoint_from_outpost_id(self, request):
        return 'outpost_id' in request.context

    def _validate_endpoint_from_arn_details_supported(self, request):
        if 'fips' in request.context['arn_details']['region']:
            raise UnsupportedS3ControlArnError(arn=request.context['arn_details']['original'], msg='Invalid ARN, FIPS region not allowed in ARN.')
        if not self._s3_config.get('use_arn_region', False):
            arn_region = request.context['arn_details']['region']
            if arn_region != self._region:
                error_msg = 'The use_arn_region configuration is disabled but received arn for "%s" when the client is configured to use "%s"' % (arn_region, self._region)
                raise UnsupportedS3ControlConfigurationError(msg=error_msg)
        request_partion = request.context['arn_details']['partition']
        if request_partion != self._partition:
            raise UnsupportedS3ControlConfigurationError(msg='Client is configured for "%s" partition, but arn provided is for "%s" partition. The client and arn partition must be the same.' % (self._partition, request_partion))
        if self._s3_config.get('use_accelerate_endpoint'):
            raise UnsupportedS3ControlConfigurationError(msg='S3 control client does not support accelerate endpoints')
        if 'outpost_name' in request.context['arn_details']:
            self._validate_outpost_redirection_valid(request)

    def _validate_outpost_redirection_valid(self, request):
        if self._s3_config.get('use_dualstack_endpoint'):
            raise UnsupportedS3ControlConfigurationError(msg='Client does not support s3 dualstack configuration when an outpost is specified.')

    def _resolve_region_from_arn_details(self, request):
        if self._s3_config.get('use_arn_region', False):
            arn_region = request.context['arn_details']['region']
            self._override_signing_region(request, arn_region)
            return arn_region
        return self._region

    def _resolve_signing_name_from_arn_details(self, request):
        arn_service = request.context['arn_details']['service']
        self._override_signing_name(request, arn_service)
        return arn_service

    def _resolve_endpoint_from_arn_details(self, request, region_name):
        new_netloc = self._resolve_netloc_from_arn_details(request, region_name)
        self._update_request_netloc(request, new_netloc)

    def _update_request_netloc(self, request, new_netloc):
        original_components = urlsplit(request.url)
        arn_details_endpoint = urlunsplit((original_components.scheme, new_netloc, original_components.path, original_components.query, ''))
        logger.debug(f'Updating URI from {request.url} to {arn_details_endpoint}')
        request.url = arn_details_endpoint

    def _resolve_netloc_from_arn_details(self, request, region_name):
        arn_details = request.context['arn_details']
        if 'outpost_name' in arn_details:
            return self._construct_outpost_endpoint(region_name)
        account = arn_details['account']
        return self._construct_s3_control_endpoint(region_name, account)

    def _is_valid_host_label(self, label):
        return self._HOST_LABEL_REGEX.match(label)

    def _validate_host_labels(self, *labels):
        for label in labels:
            if not self._is_valid_host_label(label):
                raise InvalidHostLabelError(label=label)

    def _construct_s3_control_endpoint(self, region_name, account):
        self._validate_host_labels(region_name, account)
        if self._endpoint_url:
            endpoint_url_netloc = urlsplit(self._endpoint_url).netloc
            netloc = [account, endpoint_url_netloc]
        else:
            netloc = [account, 's3-control']
            self._add_dualstack(netloc)
            dns_suffix = self._get_dns_suffix(region_name)
            netloc.extend([region_name, dns_suffix])
        return self._construct_netloc(netloc)

    def _construct_outpost_endpoint(self, region_name):
        self._validate_host_labels(region_name)
        if self._endpoint_url:
            return urlsplit(self._endpoint_url).netloc
        else:
            netloc = ['s3-outposts', region_name, self._get_dns_suffix(region_name)]
            self._add_fips(netloc)
        return self._construct_netloc(netloc)

    def _construct_netloc(self, netloc):
        return '.'.join(netloc)

    def _add_fips(self, netloc):
        if self._use_fips_endpoint:
            netloc[0] = netloc[0] + '-fips'

    def _add_dualstack(self, netloc):
        if self._s3_config.get('use_dualstack_endpoint'):
            netloc.append('dualstack')

    def _get_dns_suffix(self, region_name):
        resolved = self._endpoint_resolver.construct_endpoint('s3', region_name)
        dns_suffix = self._DEFAULT_DNS_SUFFIX
        if resolved and 'dnsSuffix' in resolved:
            dns_suffix = resolved['dnsSuffix']
        return dns_suffix

    def _override_signing_region(self, request, region_name):
        signing_context = request.context.get('signing', {})
        signing_context['region'] = region_name
        request.context['signing'] = signing_context

    def _override_signing_name(self, request, signing_name):
        signing_context = request.context.get('signing', {})
        signing_context['signing_name'] = signing_name
        request.context['signing'] = signing_context

    def _add_headers_from_arn_details(self, request):
        arn_details = request.context['arn_details']
        outpost_name = arn_details.get('outpost_name')
        if outpost_name:
            self._add_outpost_id_header(request, outpost_name)

    def _add_outpost_id_header(self, request, outpost_name):
        request.headers['x-amz-outpost-id'] = outpost_name