import traceback
from lazyops.libs import lazyload
from typing import List, Optional
class InvalidTokenException(AuthZeroException):
    """
    Invalid Token Exception
    """
    base = 'Invalid Token'
    concat_detail = True
    log_devel = True
    default_status_code = 401