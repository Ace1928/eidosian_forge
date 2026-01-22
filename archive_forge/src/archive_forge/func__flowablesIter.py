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
def _flowablesIter(self):
    for f in self._flowables:
        if isinstance(f, (list, tuple)):
            if f:
                for i, z in enumerate(f):
                    yield (i == 0 and (not isinstance(z, LIIndenter)), z)
        elif isinstance(f, ListItem):
            params = f._params
            if not params:
                for i, z in enumerate(f._flowables):
                    if isinstance(z, LIIndenter):
                        raise ValueError('LIIndenter not allowed in ListItem')
                    yield (i == 0, z)
            else:
                params = params.copy()
                value = params.pop('value', None)
                spaceBefore = params.pop('spaceBefore', None)
                spaceAfter = params.pop('spaceAfter', None)
                n = len(f._flowables) - 1
                for i, z in enumerate(f._flowables):
                    P = params.copy()
                    if not i and spaceBefore is not None:
                        P['spaceBefore'] = spaceBefore
                    if i == n and spaceAfter is not None:
                        P['spaceAfter'] = spaceAfter
                    if i:
                        value = None
                    yield (0, _LIParams(z, P, value, i == 0))
        else:
            yield (not isinstance(f, LIIndenter), f)