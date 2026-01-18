import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
def url_regex() -> Pattern[str]:
    global _url_regex_cache
    if _url_regex_cache is None:
        _url_regex_cache = re.compile(f'{_scheme_regex}{_user_info_regex}{_host_regex}{_path_regex}{_query_regex}{_fragment_regex}', re.IGNORECASE)
    return _url_regex_cache