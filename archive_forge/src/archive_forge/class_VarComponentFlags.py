from collections import namedtuple
from fontTools.misc import sstruct
from fontTools import ttLib
from fontTools import version
from fontTools.misc.transform import DecomposedTransform
from fontTools.misc.textTools import tostr, safeEval, pad
from fontTools.misc.arrayTools import updateBounds, pointInRect
from fontTools.misc.bezierTools import calcQuadraticBounds
from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.vector import Vector
from numbers import Number
from . import DefaultTable
from . import ttProgram
import sys
import struct
import array
import logging
import math
import os
from fontTools.misc import xmlWriter
from fontTools.misc.filenames import userNameToFileName
from fontTools.misc.loggingTools import deprecateFunction
from enum import IntFlag
from functools import partial
from types import SimpleNamespace
from typing import Set
class VarComponentFlags(IntFlag):
    USE_MY_METRICS = 1
    AXIS_INDICES_ARE_SHORT = 2
    UNIFORM_SCALE = 4
    HAVE_TRANSLATE_X = 8
    HAVE_TRANSLATE_Y = 16
    HAVE_ROTATION = 32
    HAVE_SCALE_X = 64
    HAVE_SCALE_Y = 128
    HAVE_SKEW_X = 256
    HAVE_SKEW_Y = 512
    HAVE_TCENTER_X = 1024
    HAVE_TCENTER_Y = 2048
    GID_IS_24BIT = 4096
    AXES_HAVE_VARIATION = 8192
    RESET_UNSPECIFIED_AXES = 16384