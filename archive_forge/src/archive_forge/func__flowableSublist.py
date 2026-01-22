import os
from copy import deepcopy, copy
from reportlab.lib.colors import gray, lightgrey
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import _baseFontName
from reportlab.lib.utils import strTypes, rl_safe_exec, annotateException
from reportlab.lib.abag import ABag
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ, overlapAttachedSpace, ignoreContainerActions, listWrapOnFakeWidth
from reportlab.lib.sequencer import _type2formatter
from reportlab.lib.styles import ListStyle
def _flowableSublist(V):
    """if it isn't a list or tuple, wrap it in a list"""
    if not isinstance(V, (list, tuple)):
        V = V is not None and [V] or []
    from reportlab.platypus.doctemplate import LCActionFlowable
    assert not [x for x in V if isinstance(x, LCActionFlowable)], 'LCActionFlowables not allowed in sublists'
    return V