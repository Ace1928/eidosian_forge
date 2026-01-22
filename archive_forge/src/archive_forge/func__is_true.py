import os
import re
import typing
from typing import Literal, Optional, Tuple
def _is_true(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.upper() in ENV_VARS_TRUE_VALUES