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
class S3RegionRedirectorv2:
    """Updated version of S3RegionRedirector for use when
    EndpointRulesetResolver is in use for endpoint resolution.

    This class is considered private and subject to abrupt breaking changes or
    removal without prior announcement. Please do not use it directly.
    """

    def __init__(self, endpoint_bridge, client, cache=None):
        self._cache = cache or {}
        self._client = weakref.proxy(client)

    def register(self, event_emitter=None):
        logger.debug('Registering S3 region redirector handler')
        emitter = event_emitter or self._client.meta.events
        emitter.register('needs-retry.s3', self.redirect_from_error)
        emitter.register('before-parameter-build.s3', self.annotate_request_context)
        emitter.register('before-endpoint-resolution.s3', self.redirect_from_cache)

    def redirect_from_error(self, request_dict, response, operation, **kwargs):
        """
        An S3 request sent to the wrong region will return an error that
        contains the endpoint the request should be sent to. This handler
        will add the redirect information to the signing context and then
        redirect the request.
        """
        if response is None:
            return
        redirect_ctx = request_dict.get('context', {}).get('s3_redirect', {})
        if ArnParser.is_arn(redirect_ctx.get('bucket')):
            logger.debug('S3 request was previously for an Accesspoint ARN, not redirecting.')
            return
        if redirect_ctx.get('redirected'):
            logger.debug('S3 request was previously redirected, not redirecting.')
            return
        error = response[1].get('Error', {})
        error_code = error.get('Code')
        response_metadata = response[1].get('ResponseMetadata', {})
        is_special_head_object = error_code in ('301', '400') and operation.name == 'HeadObject'
        is_special_head_bucket = error_code in ('301', '400') and operation.name == 'HeadBucket' and ('x-amz-bucket-region' in response_metadata.get('HTTPHeaders', {}))
        is_wrong_signing_region = error_code == 'AuthorizationHeaderMalformed' and 'Region' in error
        is_redirect_status = response[0] is not None and response[0].status_code in (301, 302, 307)
        is_permanent_redirect = error_code == 'PermanentRedirect'
        if not any([is_special_head_object, is_wrong_signing_region, is_permanent_redirect, is_special_head_bucket, is_redirect_status]):
            return
        bucket = request_dict['context']['s3_redirect']['bucket']
        client_region = request_dict['context'].get('client_region')
        new_region = self.get_bucket_region(bucket, response)
        if new_region is None:
            logger.debug('S3 client configured for region %s but the bucket %s is not in that region and the proper region could not be automatically determined.' % (client_region, bucket))
            return
        logger.debug('S3 client configured for region %s but the bucket %s is in region %s; Please configure the proper region to avoid multiple unnecessary redirects and signing attempts.' % (client_region, bucket, new_region))
        self._cache[bucket] = new_region
        ep_resolver = self._client._ruleset_resolver
        ep_info = ep_resolver.construct_endpoint(operation_model=operation, call_args=request_dict['context']['s3_redirect']['params'], request_context=request_dict['context'])
        request_dict['url'] = self.set_request_url(request_dict['url'], ep_info.url)
        request_dict['context']['s3_redirect']['redirected'] = True
        auth_schemes = ep_info.properties.get('authSchemes')
        if auth_schemes is not None:
            auth_info = ep_resolver.auth_schemes_to_signing_ctx(auth_schemes)
            auth_type, signing_context = auth_info
            request_dict['context']['auth_type'] = auth_type
            request_dict['context']['signing'] = {**request_dict['context'].get('signing', {}), **signing_context}
        return 0

    def get_bucket_region(self, bucket, response):
        """
        There are multiple potential sources for the new region to redirect to,
        but they aren't all universally available for use. This will try to
        find region from response elements, but will fall back to calling
        HEAD on the bucket if all else fails.

        :param bucket: The bucket to find the region for. This is necessary if
            the region is not available in the error response.
        :param response: A response representing a service request that failed
            due to incorrect region configuration.
        """
        service_response = response[1]
        response_headers = service_response['ResponseMetadata']['HTTPHeaders']
        if 'x-amz-bucket-region' in response_headers:
            return response_headers['x-amz-bucket-region']
        region = service_response.get('Error', {}).get('Region', None)
        if region is not None:
            return region
        try:
            response = self._client.head_bucket(Bucket=bucket)
            headers = response['ResponseMetadata']['HTTPHeaders']
        except ClientError as e:
            headers = e.response['ResponseMetadata']['HTTPHeaders']
        region = headers.get('x-amz-bucket-region', None)
        return region

    def set_request_url(self, old_url, new_endpoint, **kwargs):
        """
        Splice a new endpoint into an existing URL. Note that some endpoints
        from the the endpoint provider have a path component which will be
        discarded by this function.
        """
        return _get_new_endpoint(old_url, new_endpoint, False)

    def redirect_from_cache(self, builtins, params, **kwargs):
        """
        If a bucket name has been redirected before, it is in the cache. This
        handler will update the AWS::Region endpoint resolver builtin param
        to use the region from cache instead of the client region to avoid the
        redirect.
        """
        bucket = params.get('Bucket')
        if bucket is not None and bucket in self._cache:
            new_region = self._cache.get(bucket)
            builtins['AWS::Region'] = new_region

    def annotate_request_context(self, params, context, **kwargs):
        """Store the bucket name in context for later use when redirecting.
        The bucket name may be an access point ARN or alias.
        """
        bucket = params.get('Bucket')
        context['s3_redirect'] = {'redirected': False, 'bucket': bucket, 'params': params}