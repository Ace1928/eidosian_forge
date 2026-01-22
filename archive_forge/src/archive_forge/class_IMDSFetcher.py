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
class IMDSFetcher:
    _RETRIES_EXCEEDED_ERROR_CLS = _RetriesExceededError
    _TOKEN_PATH = 'latest/api/token'
    _TOKEN_TTL = '21600'

    def __init__(self, timeout=DEFAULT_METADATA_SERVICE_TIMEOUT, num_attempts=1, base_url=METADATA_BASE_URL, env=None, user_agent=None, config=None):
        self._timeout = timeout
        self._num_attempts = num_attempts
        if config is None:
            config = {}
        self._base_url = self._select_base_url(base_url, config)
        self._config = config
        if env is None:
            env = os.environ.copy()
        self._disabled = env.get('AWS_EC2_METADATA_DISABLED', 'false').lower() == 'true'
        self._imds_v1_disabled = config.get('ec2_metadata_v1_disabled')
        self._user_agent = user_agent
        self._session = botocore.httpsession.URLLib3Session(timeout=self._timeout, proxies=get_environ_proxies(self._base_url))

    def get_base_url(self):
        return self._base_url

    def _select_base_url(self, base_url, config):
        if config is None:
            config = {}
        requires_ipv6 = config.get('ec2_metadata_service_endpoint_mode') == 'ipv6'
        custom_metadata_endpoint = config.get('ec2_metadata_service_endpoint')
        if requires_ipv6 and custom_metadata_endpoint:
            logger.warning('Custom endpoint and IMDS_USE_IPV6 are both set. Using custom endpoint.')
        chosen_base_url = None
        if base_url != METADATA_BASE_URL:
            chosen_base_url = base_url
        elif custom_metadata_endpoint:
            chosen_base_url = custom_metadata_endpoint
        elif requires_ipv6:
            chosen_base_url = METADATA_BASE_URL_IPv6
        else:
            chosen_base_url = METADATA_BASE_URL
        logger.debug('IMDS ENDPOINT: %s' % chosen_base_url)
        if not is_valid_uri(chosen_base_url):
            raise InvalidIMDSEndpointError(endpoint=chosen_base_url)
        return chosen_base_url

    def _construct_url(self, path):
        sep = ''
        if self._base_url and (not self._base_url.endswith('/')):
            sep = '/'
        return f'{self._base_url}{sep}{path}'

    def _fetch_metadata_token(self):
        self._assert_enabled()
        url = self._construct_url(self._TOKEN_PATH)
        headers = {'x-aws-ec2-metadata-token-ttl-seconds': self._TOKEN_TTL}
        self._add_user_agent(headers)
        request = botocore.awsrequest.AWSRequest(method='PUT', url=url, headers=headers)
        for i in range(self._num_attempts):
            try:
                response = self._session.send(request.prepare())
                if response.status_code == 200:
                    return response.text
                elif response.status_code in (404, 403, 405):
                    return None
                elif response.status_code in (400,):
                    raise BadIMDSRequestError(request)
            except ReadTimeoutError:
                return None
            except RETRYABLE_HTTP_ERRORS as e:
                logger.debug('Caught retryable HTTP exception while making metadata service request to %s: %s', url, e, exc_info=True)
            except HTTPClientError as e:
                if isinstance(e.kwargs.get('error'), LocationParseError):
                    raise InvalidIMDSEndpointError(endpoint=url, error=e)
                else:
                    raise
        return None

    def _get_request(self, url_path, retry_func, token=None):
        """Make a get request to the Instance Metadata Service.

        :type url_path: str
        :param url_path: The path component of the URL to make a get request.
            This arg is appended to the base_url that was provided in the
            initializer.

        :type retry_func: callable
        :param retry_func: A function that takes the response as an argument
             and determines if it needs to retry. By default empty and non
             200 OK responses are retried.

        :type token: str
        :param token: Metadata token to send along with GET requests to IMDS.
        """
        self._assert_enabled()
        if not token:
            self._assert_v1_enabled()
        if retry_func is None:
            retry_func = self._default_retry
        url = self._construct_url(url_path)
        headers = {}
        if token is not None:
            headers['x-aws-ec2-metadata-token'] = token
        self._add_user_agent(headers)
        for i in range(self._num_attempts):
            try:
                request = botocore.awsrequest.AWSRequest(method='GET', url=url, headers=headers)
                response = self._session.send(request.prepare())
                if not retry_func(response):
                    return response
            except RETRYABLE_HTTP_ERRORS as e:
                logger.debug('Caught retryable HTTP exception while making metadata service request to %s: %s', url, e, exc_info=True)
        raise self._RETRIES_EXCEEDED_ERROR_CLS()

    def _add_user_agent(self, headers):
        if self._user_agent is not None:
            headers['User-Agent'] = self._user_agent

    def _assert_enabled(self):
        if self._disabled:
            logger.debug('Access to EC2 metadata has been disabled.')
            raise self._RETRIES_EXCEEDED_ERROR_CLS()

    def _assert_v1_enabled(self):
        if self._imds_v1_disabled:
            raise MetadataRetrievalError(error_msg='Unable to retrieve token for use in IMDSv2 call and IMDSv1 has been disabled')

    def _default_retry(self, response):
        return self._is_non_ok_response(response) or self._is_empty(response)

    def _is_non_ok_response(self, response):
        if response.status_code != 200:
            self._log_imds_response(response, 'non-200', log_body=True)
            return True
        return False

    def _is_empty(self, response):
        if not response.content:
            self._log_imds_response(response, 'no body', log_body=True)
            return True
        return False

    def _log_imds_response(self, response, reason_to_log, log_body=False):
        statement = 'Metadata service returned %s response with status code of %s for url: %s'
        logger_args = [reason_to_log, response.status_code, response.url]
        if log_body:
            statement += ', content body: %s'
            logger_args.append(response.content)
        logger.debug(statement, *logger_args)