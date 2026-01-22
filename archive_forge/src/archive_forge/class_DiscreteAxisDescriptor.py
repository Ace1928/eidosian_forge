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
class DiscreteAxisDescriptor(AbstractAxisDescriptor):
    """Container for discrete axis data.

    Use this for axes that do not interpolate. The main difference from a
    continuous axis is that a continuous axis has a ``minimum`` and ``maximum``,
    while a discrete axis has a list of ``values``.

    Example: an Italic axis with 2 stops, Roman and Italic, that are not
    compatible. The axis still allows to bind together the full font family,
    which is useful for the STAT table, however it can't become a variation
    axis in a VF.

    .. code:: python

        a2 = DiscreteAxisDescriptor()
        a2.values = [0, 1]
        a2.default = 0
        a2.name = "Italic"
        a2.tag = "ITAL"
        a2.labelNames['fr'] = "Italique"
        a2.map = [(0, 0), (1, -11)]
        a2.axisOrdering = 2
        a2.axisLabels = [
            AxisLabelDescriptor(name="Roman", userValue=0, elidable=True)
        ]
        doc.addAxis(a2)

    .. versionadded:: 5.0
    """
    flavor = 'axis'
    _attrs = ('tag', 'name', 'values', 'default', 'map', 'axisOrdering', 'axisLabels')

    def __init__(self, *, tag=None, name=None, labelNames=None, values=None, default=None, hidden=False, map=None, axisOrdering=None, axisLabels=None):
        super().__init__(tag=tag, name=name, labelNames=labelNames, hidden=hidden, map=map, axisOrdering=axisOrdering, axisLabels=axisLabels)
        self.default: float = default
        'The default value for this axis, i.e. when a new location is\n        created, this is the value this axis will get in user space.\n\n        However, this default value is less important than in continuous axes:\n\n        -  it doesn\'t define the "neutral" version of outlines from which\n           deltas would apply, as this axis does not interpolate.\n        -  it doesn\'t provide the reference glyph set for the designspace, as\n           fonts at each value can have different glyph sets.\n        '
        self.values: List[float] = values or []
        'List of possible values for this axis. Contrary to continuous axes,\n        only the values in this list can be taken by the axis, nothing in-between.\n        '

    def map_forward(self, value):
        """Maps value from axis mapping's input to output.

        Returns value unchanged if no mapping entry is found.

        Note: for discrete axes, each value must have its mapping entry, if
        you intend that value to be mapped.
        """
        return next((v for k, v in self.map if k == value), value)

    def map_backward(self, value):
        """Maps value from axis mapping's output to input.

        Returns value unchanged if no mapping entry is found.

        Note: for discrete axes, each value must have its mapping entry, if
        you intend that value to be mapped.
        """
        if isinstance(value, tuple):
            value = value[0]
        return next((k for k, v in self.map if v == value), value)