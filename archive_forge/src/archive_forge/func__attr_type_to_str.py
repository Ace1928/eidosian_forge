import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def _attr_type_to_str(attr_type: int) -> str:
    """Convert AttributeProto type to string.

    Args:
        attr_type: AttributeProto type.

    Returns:
        String representing the supplied attr_type.
    """
    if attr_type in AttributeProto.AttributeType.values():
        return _ATTRIBUTE_TYPE_TO_STR[attr_type]
    return AttributeProto.AttributeType.keys()[0]