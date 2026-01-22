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
class S3RegionRedirector:
    """This handler has been replaced by S3RegionRedirectorv2. The original
    version remains in place for any third-party libraries that import it.
    """

    def __init__(self, endpoint_bridge, client, cache=None):
        self._endpoint_resolver = endpoint_bridge
        self._cache = cache
        if self._cache is None:
            self._cache = {}
        self._client = weakref.proxy(client)
        warnings.warn('The S3RegionRedirector class has been deprecated for a new internal replacement. A future version of botocore may remove this class.', category=FutureWarning)

    def register(self, event_emitter=None):
        emitter = event_emitter or self._client.meta.events
        emitter.register('needs-retry.s3', self.redirect_from_error)
        emitter.register('before-call.s3', self.set_request_url)
        emitter.register('before-parameter-build.s3', self.redirect_from_cache)

    def redirect_from_error(self, request_dict, response, operation, **kwargs):
        """
        An S3 request sent to the wrong region will return an error that
        contains the endpoint the request should be sent to. This handler
        will add the redirect information to the signing context and then
        redirect the request.
        """
        if response is None:
            return
        if self._is_s3_accesspoint(request_dict.get('context', {})):
            logger.debug('S3 request was previously to an accesspoint, not redirecting.')
            return
        if request_dict.get('context', {}).get('s3_redirected'):
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
        bucket = request_dict['context']['signing']['bucket']
        client_region = request_dict['context'].get('client_region')
        new_region = self.get_bucket_region(bucket, response)
        if new_region is None:
            logger.debug('S3 client configured for region %s but the bucket %s is not in that region and the proper region could not be automatically determined.' % (client_region, bucket))
            return
        logger.debug('S3 client configured for region %s but the bucket %s is in region %s; Please configure the proper region to avoid multiple unnecessary redirects and signing attempts.' % (client_region, bucket, new_region))
        endpoint = self._endpoint_resolver.resolve('s3', new_region)
        endpoint = endpoint['endpoint_url']
        signing_context = {'region': new_region, 'bucket': bucket, 'endpoint': endpoint}
        request_dict['context']['signing'] = signing_context
        self._cache[bucket] = signing_context
        self.set_request_url(request_dict, request_dict['context'])
        request_dict['context']['s3_redirected'] = True
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

    def set_request_url(self, params, context, **kwargs):
        endpoint = context.get('signing', {}).get('endpoint', None)
        if endpoint is not None:
            params['url'] = _get_new_endpoint(params['url'], endpoint, False)

    def redirect_from_cache(self, params, context, **kwargs):
        """
        This handler retrieves a given bucket's signing context from the cache
        and adds it into the request context.
        """
        if self._is_s3_accesspoint(context):
            return
        bucket = params.get('Bucket')
        signing_context = self._cache.get(bucket)
        if signing_context is not None:
            context['signing'] = signing_context
        else:
            context['signing'] = {'bucket': bucket}

    def _is_s3_accesspoint(self, context):
        return 's3_accesspoint' in context