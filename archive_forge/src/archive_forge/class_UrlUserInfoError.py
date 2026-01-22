from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class UrlUserInfoError(UrlError):
    code = 'url.userinfo'
    msg_template = 'userinfo required in URL but missing'