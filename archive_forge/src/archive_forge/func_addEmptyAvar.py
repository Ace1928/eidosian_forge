from fontTools.ttLib import newTable
from fontTools.ttLib.tables._f_v_a_r import Axis as fvarAxis
from fontTools.pens.areaPen import AreaPen
from fontTools.pens.basePen import NullPen
from fontTools.pens.statisticsPen import StatisticsPen
from fontTools.varLib.models import piecewiseLinearMap, normalizeValue
from fontTools.misc.cliTools import makeOutputFileName
import math
import logging
from pprint import pformat
def addEmptyAvar(font):
    """Add an empty `avar` table to the font."""
    font['avar'] = avar = newTable('avar')
    for axis in fvar.axes:
        avar.segments[axis.axisTag] = {}