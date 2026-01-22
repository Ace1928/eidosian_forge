import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
@staticmethod
def _match_url(url: str) -> Optional[Match[str]]:
    return multi_host_url_regex().match(url)