from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
class FormatSerSchema(TypedDict, total=False):
    type: Required[Literal['format']]
    formatting_string: Required[str]
    when_used: WhenUsed