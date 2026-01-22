import traceback
from lazyops.libs import lazyload
from typing import List, Optional
class NoTokenException(AuthZeroException):
    """
    No Token Exception
    """
    base = 'Not Authorized: no auth token found'
    concat_detail = True
    log_devel = True
    default_status_code = 401