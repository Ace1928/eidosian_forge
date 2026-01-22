from __future__ import absolute_import
import io
import logging
import warnings
from ..exceptions import (
from ..packages.six.moves.urllib.parse import urljoin
from ..request import RequestMethods
from ..response import HTTPResponse
from ..util.retry import Retry
from ..util.timeout import Timeout
from . import _appengine_environ
class AppEngineManager(RequestMethods):
    """
    Connection manager for Google App Engine sandbox applications.

    This manager uses the URLFetch service directly instead of using the
    emulated httplib, and is subject to URLFetch limitations as described in
    the App Engine documentation `here
    <https://cloud.google.com/appengine/docs/python/urlfetch>`_.

    Notably it will raise an :class:`AppEnginePlatformError` if:
        * URLFetch is not available.
        * If you attempt to use this on App Engine Flexible, as full socket
          support is available.
        * If a request size is more than 10 megabytes.
        * If a response size is more than 32 megabytes.
        * If you use an unsupported request method such as OPTIONS.

    Beyond those cases, it will raise normal urllib3 errors.
    """

    def __init__(self, headers=None, retries=None, validate_certificate=True, urlfetch_retries=True):
        if not urlfetch:
            raise AppEnginePlatformError('URLFetch is not available in this environment.')
        warnings.warn('urllib3 is using URLFetch on Google App Engine sandbox instead of sockets. To use sockets directly instead of URLFetch see https://urllib3.readthedocs.io/en/1.26.x/reference/urllib3.contrib.html.', AppEnginePlatformWarning)
        RequestMethods.__init__(self, headers)
        self.validate_certificate = validate_certificate
        self.urlfetch_retries = urlfetch_retries
        self.retries = retries or Retry.DEFAULT

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def urlopen(self, method, url, body=None, headers=None, retries=None, redirect=True, timeout=Timeout.DEFAULT_TIMEOUT, **response_kw):
        retries = self._get_retries(retries, redirect)
        try:
            follow_redirects = redirect and retries.redirect != 0 and retries.total
            response = urlfetch.fetch(url, payload=body, method=method, headers=headers or {}, allow_truncated=False, follow_redirects=self.urlfetch_retries and follow_redirects, deadline=self._get_absolute_timeout(timeout), validate_certificate=self.validate_certificate)
        except urlfetch.DeadlineExceededError as e:
            raise TimeoutError(self, e)
        except urlfetch.InvalidURLError as e:
            if 'too large' in str(e):
                raise AppEnginePlatformError('URLFetch request too large, URLFetch only supports requests up to 10mb in size.', e)
            raise ProtocolError(e)
        except urlfetch.DownloadError as e:
            if 'Too many redirects' in str(e):
                raise MaxRetryError(self, url, reason=e)
            raise ProtocolError(e)
        except urlfetch.ResponseTooLargeError as e:
            raise AppEnginePlatformError('URLFetch response too large, URLFetch only supportsresponses up to 32mb in size.', e)
        except urlfetch.SSLCertificateError as e:
            raise SSLError(e)
        except urlfetch.InvalidMethodError as e:
            raise AppEnginePlatformError('URLFetch does not support method: %s' % method, e)
        http_response = self._urlfetch_response_to_http_response(response, retries=retries, **response_kw)
        redirect_location = redirect and http_response.get_redirect_location()
        if redirect_location:
            if self.urlfetch_retries and retries.raise_on_redirect:
                raise MaxRetryError(self, url, 'too many redirects')
            else:
                if http_response.status == 303:
                    method = 'GET'
                try:
                    retries = retries.increment(method, url, response=http_response, _pool=self)
                except MaxRetryError:
                    if retries.raise_on_redirect:
                        raise MaxRetryError(self, url, 'too many redirects')
                    return http_response
                retries.sleep_for_retry(http_response)
                log.debug('Redirecting %s -> %s', url, redirect_location)
                redirect_url = urljoin(url, redirect_location)
                return self.urlopen(method, redirect_url, body, headers, retries=retries, redirect=redirect, timeout=timeout, **response_kw)
        has_retry_after = bool(http_response.getheader('Retry-After'))
        if retries.is_retry(method, http_response.status, has_retry_after):
            retries = retries.increment(method, url, response=http_response, _pool=self)
            log.debug('Retry: %s', url)
            retries.sleep(http_response)
            return self.urlopen(method, url, body=body, headers=headers, retries=retries, redirect=redirect, timeout=timeout, **response_kw)
        return http_response

    def _urlfetch_response_to_http_response(self, urlfetch_resp, **response_kw):
        if is_prod_appengine():
            content_encoding = urlfetch_resp.headers.get('content-encoding')
            if content_encoding == 'deflate':
                del urlfetch_resp.headers['content-encoding']
        transfer_encoding = urlfetch_resp.headers.get('transfer-encoding')
        if transfer_encoding == 'chunked':
            encodings = transfer_encoding.split(',')
            encodings.remove('chunked')
            urlfetch_resp.headers['transfer-encoding'] = ','.join(encodings)
        original_response = HTTPResponse(body=io.BytesIO(urlfetch_resp.content), msg=urlfetch_resp.header_msg, headers=urlfetch_resp.headers, status=urlfetch_resp.status_code, **response_kw)
        return HTTPResponse(body=io.BytesIO(urlfetch_resp.content), headers=urlfetch_resp.headers, status=urlfetch_resp.status_code, original_response=original_response, **response_kw)

    def _get_absolute_timeout(self, timeout):
        if timeout is Timeout.DEFAULT_TIMEOUT:
            return None
        if isinstance(timeout, Timeout):
            if timeout._read is not None or timeout._connect is not None:
                warnings.warn('URLFetch does not support granular timeout settings, reverting to total or default URLFetch timeout.', AppEnginePlatformWarning)
            return timeout.total
        return timeout

    def _get_retries(self, retries, redirect):
        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, redirect=redirect, default=self.retries)
        if retries.connect or retries.read or retries.redirect:
            warnings.warn('URLFetch only supports total retries and does not recognize connect, read, or redirect retry parameters.', AppEnginePlatformWarning)
        return retries