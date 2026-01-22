from __future__ import annotations
import math
from collections import namedtuple
from functools import partial
from typing import (
import numpy as np
import param
from bokeh.models import FlexBox as BkFlexBox, GridBox as BkGridBox
from ..io.document import freeze_doc
from ..io.model import hold
from ..io.resources import CDN_DIST
from .base import (
def _compute_css_classes(self, children):
    equal_widths, equal_heights = (True, True)
    for child, _, _, _, _ in children:
        if child.sizing_mode and (child.sizing_mode.endswith('_both') or child.sizing_mode.endswith('_width')):
            equal_widths &= True
        else:
            equal_widths = False
        if child.sizing_mode and (child.sizing_mode.endswith('_both') or child.sizing_mode.endswith('_height')):
            equal_heights &= True
        else:
            equal_heights = False
    css_classes = []
    if equal_widths:
        css_classes.append('equal-width')
    if equal_heights:
        css_classes.append('equal-height')
    return css_classes