from collections import namedtuple
from collections.abc import Sequence
import numbers
import math
import re
import warnings
from traitlets import (
from ipywidgets import widget_serialization
from ipydatawidgets import DataUnion, NDArrayWidget, shape_constraints
import numpy as np
def _serialize_uninitialized(value, owner):
    if isinstance(value, Uninitialized):
        return 'uninitialized'
    return _widget_to_json(value, owner)