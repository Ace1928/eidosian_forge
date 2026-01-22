from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
def _changer(self, obj):
    """
        When implemented this method should return a dictionary of
        original attribute values so that a future self(False,obj)
        can restore them.
        """
    raise RuntimeError('Abstract method _changer called')