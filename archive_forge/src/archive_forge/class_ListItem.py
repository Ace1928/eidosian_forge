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
class ListItem:

    def __init__(self, flowables, style=None, **kwds):
        if not isinstance(flowables, (list, tuple)):
            flowables = (flowables,)
        self._flowables = flowables
        params = self._params = {}
        if style:
            if not isinstance(style, ListStyle):
                raise ValueError('%s style argument (%r) not a ListStyle' % (self.__class__.__name__, style))
            self._style = style
        for k in ListStyle.defaults:
            if k in kwds:
                v = kwds.get(k)
            elif style:
                v = getattr(style, k)
            else:
                continue
            params[k] = v
        for k in ('value', 'spaceBefore', 'spaceAfter'):
            v = kwds.get(k, getattr(style, k, None))
            if v is not None:
                params[k] = v