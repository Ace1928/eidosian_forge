import copy
import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from ..errors import Errors
Check if an extension attribute is writable.
    ext (tuple): The (default, getter, setter, method) tuple available  via
        {Doc,Span,Token}.get_extension.
    RETURNS (bool): Whether the attribute is writable.
    