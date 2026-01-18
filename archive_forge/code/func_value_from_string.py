from math import nan
from typing import Any, Callable, Dict, Optional, Union
from ..language import (
from ..pyutils import inspect, Undefined
def value_from_string(value_node: Union[BooleanValueNode, EnumValueNode, StringValueNode], _variables: Any) -> Any:
    return value_node.value