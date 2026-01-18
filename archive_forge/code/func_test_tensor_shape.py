import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
def test_tensor_shape(op_name: str) -> str:
    if op_name in test_tensor_shape_json:
        return test_tensor_shape_json[op_name]
    else:
        return ''