from __future__ import annotations
import logging
import sys
from typing import Any
from typing import Optional
from typing import overload
from typing import Set
from typing import Type
from typing import TypeVar
from typing import Union
from .util import py311
from .util import py38
from .util.typing import Literal
def _should_log_debug(self) -> bool:
    return self.logger.isEnabledFor(logging.DEBUG)