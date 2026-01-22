import json
import logging
from datetime import datetime, timezone
from logging import Logger
from types import TracebackType
from typing import Any, Optional
from requests import Response
from requests.models import CaseInsensitiveDict
from requests.utils import get_encoding_from_headers
from typing_extensions import Self
from urllib3 import Retry
from urllib3.connectionpool import ConnectionPool
from urllib3.exceptions import MaxRetryError
from urllib3.response import HTTPResponse
from github.GithubException import GithubException
from github.Requester import Requester
class GithubRetry(Retry):
    """
    A Github-specific implementation of `urllib3.Retry`

    This retries 403 responses if they are retry-able. Github requests are retry-able when
    the response provides a `"Retry-After"` header, or the content indicates a rate limit error.

    By default, response codes 403, and 500 up to 599 are retried. This can be configured
    via the `status_forcelist` argument.

    By default, all methods defined in `Retry.DEFAULT_ALLOWED_METHODS` are retried, plus GET and POST.
    This can be configured via the `allowed_methods` argument.
    """
    __logger: Optional[Logger] = None
    __datetime = datetime

    def __init__(self, secondary_rate_wait: float=DEFAULT_SECONDARY_RATE_WAIT, **kwargs: Any) -> None:
        """
        :param secondary_rate_wait: seconds to wait before retrying secondary rate limit errors
        :param kwargs: see urllib3.Retry for more arguments
        """
        self.secondary_rate_wait = secondary_rate_wait
        kwargs['status_forcelist'] = kwargs.get('status_forcelist', list(range(500, 600))) + [403]
        kwargs['allowed_methods'] = kwargs.get('allowed_methods', Retry.DEFAULT_ALLOWED_METHODS.union({'GET', 'POST'}))
        super().__init__(**kwargs)

    def new(self, **kw: Any) -> Self:
        kw.update(dict(secondary_rate_wait=self.secondary_rate_wait))
        return super().new(**kw)

    def increment(self, method: Optional[str]=None, url: Optional[str]=None, response: Optional[HTTPResponse]=None, error: Optional[Exception]=None, _pool: Optional[ConnectionPool]=None, _stacktrace: Optional[TracebackType]=None) -> Retry:
        if response:
            if response.status == 403:
                self.__log(logging.INFO, f'Request {method} {url} failed with {response.status}: {response.reason}')
                if 'Retry-After' in response.headers:
                    self.__log(logging.INFO, f'Retrying after {response.headers.get('Retry-After')} seconds')
                else:
                    content = response.reason
                    try:
                        content = self.get_content(response, url)
                        content = json.loads(content)
                        message = content.get('message')
                    except Exception as e:
                        try:
                            raise RuntimeError('Failed to inspect response message') from e
                        except RuntimeError as e:
                            raise GithubException(response.status, content, response.headers) from e
                    try:
                        if Requester.isRateLimitError(message):
                            rate_type = 'primary' if Requester.isPrimaryRateLimitError(message) else 'secondary'
                            self.__log(logging.DEBUG, f'Response body indicates retry-able {rate_type} rate limit error: {message}')
                            retry = super().increment(method, url, response, error, _pool, _stacktrace)
                            backoff = 0.0
                            if Requester.isPrimaryRateLimitError(message):
                                if 'X-RateLimit-Reset' in response.headers:
                                    value = response.headers.get('X-RateLimit-Reset')
                                    if value and value.isdigit():
                                        reset = self.__datetime.fromtimestamp(int(value), timezone.utc)
                                        delta = reset - self.__datetime.now(timezone.utc)
                                        resetBackoff = delta.total_seconds()
                                        if resetBackoff > 0:
                                            self.__log(logging.DEBUG, f'Reset occurs in {str(delta)} ({value} / {reset})')
                                        backoff = resetBackoff + 1
                            else:
                                backoff = self.secondary_rate_wait
                            retry_backoff = retry.get_backoff_time()
                            if retry_backoff > backoff:
                                if backoff > 0:
                                    self.__log(logging.DEBUG, f'Retry backoff of {retry_backoff}s exceeds required rate limit backoff of {backoff}s'.replace('.0s', 's'))
                                backoff = retry_backoff

                            def get_backoff_time() -> float:
                                return backoff
                            self.__log(logging.INFO, f'Setting next backoff to {backoff}s'.replace('.0s', 's'))
                            retry.get_backoff_time = get_backoff_time
                            return retry
                        self.__log(logging.DEBUG, 'Response message does not indicate retry-able error')
                        raise Requester.createException(response.status, response.headers, content)
                    except (MaxRetryError, GithubException):
                        raise
                    except Exception as e:
                        try:
                            raise RuntimeError('Failed to determine retry backoff') from e
                        except RuntimeError as e:
                            raise GithubException(response.status, content, response.headers) from e
                    raise GithubException(response.status, content, response.headers)
        return super().increment(method, url, response, error, _pool, _stacktrace)

    @staticmethod
    def get_content(resp: HTTPResponse, url: str) -> bytes:
        response = Response()
        response.status_code = getattr(resp, 'status', None)
        response.headers = CaseInsensitiveDict(getattr(resp, 'headers', {}))
        response.encoding = get_encoding_from_headers(response.headers)
        response.raw = resp
        response.reason = response.raw.reason
        response.url = url
        return response.content

    def __log(self, level: int, message: str, **kwargs: Any) -> None:
        if self.__logger is None:
            self.__logger = logging.getLogger(__name__)
        if self.__logger.isEnabledFor(level):
            self.__logger.log(level, message, **kwargs)