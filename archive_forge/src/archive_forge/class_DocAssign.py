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
class DocAssign(NullDraw):
    """At wrap time this flowable evaluates var=expr in the doctemplate namespace"""
    _ZEROSIZE = 1

    def __init__(self, var, expr, life='forever'):
        Flowable.__init__(self)
        self.args = (var, expr, life)

    def funcWrap(self, aW, aH):
        NS = self._doctemplateAttr('_nameSpace')
        NS.update(dict(availableWidth=aW, availableHeight=aH))
        try:
            return self.func()
        finally:
            for k in ('availableWidth', 'availableHeight'):
                try:
                    del NS[k]
                except:
                    pass

    def func(self):
        return self._doctemplateAttr('d' + self.__class__.__name__[1:])(*self.args)

    def wrap(self, aW, aH):
        self.funcWrap(aW, aH)
        return (0, 0)