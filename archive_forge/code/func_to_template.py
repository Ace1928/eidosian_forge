import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
def to_template(data: Any) -> TuningParametersTemplate:
    """Convert an oject to ``TuningParametersTemplate``

    :param data: data object (``dict`` or ``TuningParametersTemplate``
        or ``str`` (encoded string))
    :return: the template object
    """
    if isinstance(data, TuningParametersTemplate):
        return data
    if isinstance(data, dict):
        return TuningParametersTemplate(data)
    if isinstance(data, str):
        return TuningParametersTemplate.decode(data)
    raise ValueError(f"can't convert to template: {data}")