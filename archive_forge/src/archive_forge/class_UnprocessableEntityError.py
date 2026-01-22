from __future__ import annotations
from typing import Any, Optional, cast
from typing_extensions import Literal
import httpx
from ._utils import is_dict
class UnprocessableEntityError(APIStatusError):
    status_code: Literal[422] = 422