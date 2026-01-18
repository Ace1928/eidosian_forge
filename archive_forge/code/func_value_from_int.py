from math import nan
from typing import Any, Callable, Dict, Optional, Union
from ..language import (
from ..pyutils import inspect, Undefined
def value_from_int(value_node: IntValueNode, _variables: Any) -> Any:
    try:
        return int(value_node.value)
    except ValueError:
        return nan