from math import nan
from typing import Any, Callable, Dict, Optional, Union
from ..language import (
from ..pyutils import inspect, Undefined
def value_from_null(_value_node: NullValueNode, _variables: Any) -> Any:
    return None