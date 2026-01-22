from reportlab.platypus.flowables import Flowable, Preformatted
from reportlab import rl_config
from reportlab.lib.styles import PropertySet, ParagraphStyle, _baseFontName
from reportlab.lib import colors
from reportlab.lib.utils import annotateException, IdentStr, flatten, isStr, asNative, strTypes, __UNSET__
from reportlab.lib.validators import isListOfNumbersOrNone
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.abag import ABag as CellFrame
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus.doctemplate import Indenter, NullActionFlowable
from reportlab.platypus.flowables import LIIndenter
from collections import namedtuple
def _convert2int(value, map, low, high, name, cmd):
    """private converter tries map(value) low<=int(value)<=high or finally an error"""
    try:
        return map[value]
    except KeyError:
        try:
            ivalue = int(value)
            if low <= ivalue <= high:
                return ivalue
        except:
            pass
    raise ValueError(f'Bad {name} value {value} in {cmd!a}')