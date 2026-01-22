import time
from email.utils import mktime_tz, parsedate_tz
class RateLimitReachedError(BaseHTTPError):
    """
    HTTP 429 - Rate limit: you've sent too many requests for this time period.
    """
    code = 429
    message = '%s Rate limit exceeded' % code

    def __init__(self, *args, **kwargs):
        headers = kwargs.pop('headers', None)
        super().__init__(self.code, self.message, headers)
        if self.headers is not None:
            self.retry_after = float(self.headers.get('retry-after', 0))
        else:
            self.retry_after = 0