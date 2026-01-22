import traceback
from lazyops.libs import lazyload
from typing import List, Optional
class NoAPIKeyException(AuthZeroException):
    """
    No API Key Exception
    """
    base = 'Not Authorized'
    concat_detail = True
    log_devel = True
    default_status_code = 401

    def __init__(self, key: str='x-api-key', detail: str=None, **kwargs):
        """
        Constructor
        """
        self.base += f' no `{key}` found'
        super().__init__(detail=detail, **kwargs)