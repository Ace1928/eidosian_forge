from __future__ import annotations
import collections
import copy
import itertools
import math
import os
import posixpath
from io import BytesIO, StringIO
from textwrap import indent
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union, cast
from fontTools.misc import etree as ET
from fontTools.misc import plistlib
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr
class RangeAxisSubsetDescriptor(SimpleDescriptor):
    """Subset of a continuous axis to include in a variable font.

    .. versionadded:: 5.0
    """
    flavor = 'axis-subset'
    _attrs = ('name', 'userMinimum', 'userDefault', 'userMaximum')

    def __init__(self, *, name, userMinimum=-math.inf, userDefault=None, userMaximum=math.inf):
        self.name: str = name
        'Name of the :class:`AxisDescriptor` to subset.'
        self.userMinimum: float = userMinimum
        'New minimum value of the axis in the target variable font.\n        If not specified, assume the same minimum value as the full axis.\n        (default = ``-math.inf``)\n        '
        self.userDefault: Optional[float] = userDefault
        'New default value of the axis in the target variable font.\n        If not specified, assume the same default value as the full axis.\n        (default = ``None``)\n        '
        self.userMaximum: float = userMaximum
        'New maximum value of the axis in the target variable font.\n        If not specified, assume the same maximum value as the full axis.\n        (default = ``math.inf``)\n        '