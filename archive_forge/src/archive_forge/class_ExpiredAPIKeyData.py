import traceback
from lazyops.libs import lazyload
from typing import List, Optional
class ExpiredAPIKeyData(AuthZeroException):
    """
    Expired API Key Data
    """
    base = 'Expired API Key Data'
    concat_detail = True
    log_devel = True
    default_status_code = 403