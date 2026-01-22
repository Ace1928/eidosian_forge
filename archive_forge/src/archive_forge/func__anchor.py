import logging
import re
from io import StringIO
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser
def _anchor(self, adjustment):
    adv, dx, dy, adv_adjust_by, dx_adjust_by, dy_adjust_by = adjustment
    assert not adv_adjust_by
    dx_device = dx_adjust_by and dx_adjust_by.items() or None
    dy_device = dy_adjust_by and dy_adjust_by.items() or None
    return ast.Anchor(dx or 0, dy or 0, xDeviceTable=dx_device or None, yDeviceTable=dy_device or None)