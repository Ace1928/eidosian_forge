import traceback
from lazyops.libs import lazyload
from typing import List, Optional
class InvalidAPIKeyPrefixException(AuthZeroException):
    """
    Invalid API Key Prefix Exception
    """
    base = 'Invalid API Key Prefix. Your API Key may be deprecated. Please regenerate your API Key by logging in again'
    concat_detail = True
    log_devel = False
    default_status_code = 401