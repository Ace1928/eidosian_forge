import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def add_shape_compute_mapping(operator_schema: str, func: Callable):
    global shape_compute_graph_mapping
    shape_compute_graph_mapping[operator_schema] = process_func(func)