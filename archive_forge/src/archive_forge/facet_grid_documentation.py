from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from .._utils import add_margins, cross_join, join_keys, match, ninteraction
from ..exceptions import PlotnineError
from .facet import (
from .strips import Strips, strip

    Convert a str specification to a list spec

    e.g.
    'a' -> ['a']
    'a + b' -> ['a', 'b']
    '.' -> []
    '' -> []
    