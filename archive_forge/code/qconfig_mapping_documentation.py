from __future__ import annotations
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple, Union, List
import torch
from .fake_quantize import (
from .observer import (
from .qconfig import (

        Create a ``QConfigMapping`` from a dictionary with the following keys (all optional):

            "" (for global QConfig)

            "object_type"

            "module_name_regex"

            "module_name"

            "module_name_object_type_order"

        The values of this dictionary are expected to be lists of tuples.
        