import os, sys
from math import pi, cos, sin, sqrt, radians, floor
from reportlab.platypus import Flowable
from reportlab.rl_config import shapeChecking, verbose, defaultGraphicsFontName as _baseGFontName, _unset_, decimalSymbol
from reportlab.lib import logger
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.utils import isSeq, asBytes
from reportlab.lib.attrmap import *
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.fonts import tt2ps
from reportlab.pdfgen.canvas import FILL_EVEN_ODD, FILL_NON_ZERO
from . transform import *
class _DrawingEditorMixin:
    """This is a mixin to provide functionality for edited drawings"""

    def _add(self, obj, value, name=None, validate=None, desc=None, pos=None):
        """
        effectively setattr(obj,name,value), but takes care of things with _attrMaps etc
        """
        ivc = isValidChild(value)
        if name and hasattr(obj, '_attrMap'):
            if '_attrMap' not in obj.__dict__:
                obj._attrMap = obj._attrMap.clone()
            if ivc and validate is None:
                validate = isValidChild
            obj._attrMap[name] = AttrMapValue(validate, desc)
        if hasattr(obj, 'add') and ivc:
            if pos:
                obj.insert(pos, value, name)
            else:
                obj.add(value, name)
        elif name:
            setattr(obj, name, value)
        else:
            raise ValueError("Can't add, need name")