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
class ListFlowable(_Container, Flowable):
    _numberStyles = '1aAiI'

    def __init__(self, flowables, start=None, style=None, **kwds):
        self._flowables = flowables
        if style:
            if not isinstance(style, ListStyle):
                raise ValueError('%s style argument not a ListStyle' % self.__class__.__name__)
            self.style = style
        for k, v in ListStyle.defaults.items():
            setattr(self, '_' + k, kwds.get(k, getattr(style, k, v)))
        for k in ('spaceBefore', 'spaceAfter'):
            v = kwds.get(k, getattr(style, k, None))
            if v is not None:
                setattr(self, k, v)
        auto = False
        if start is None:
            start = getattr(self, '_start', None)
            if start is None:
                if self._bulletType == 'bullet':
                    start = 'bulletchar'
                    auto = True
                else:
                    start = self._bulletType
                    auto = True
        if self._bulletType != 'bullet':
            if auto:
                for v in start:
                    if v not in self._numberStyles:
                        raise ValueError('invalid start=%r or bullettype=%r' % (start, self._bulletType))
            else:
                for v in self._bulletType:
                    if v not in self._numberStyles:
                        raise ValueError('invalid bullettype=%r' % self._bulletType)
        self._start = start
        self._auto = auto or isinstance(start, (list, tuple))
        self._list_content = None
        self._dims = None
        self._caption = kwds.pop('caption', None)

    @property
    def _content(self):
        if self._list_content is None:
            self._list_content = self._getContent()
            del self._flowables
        return self._list_content

    def wrap(self, aW, aH):
        if self._dims != aW:
            self.width, self.height = _listWrapOn(self._content, aW, self.canv)
            self._dims = aW
        return (self.width, self.height)

    def split(self, aW, aH):
        return self._content

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

    def _makeLIIndenter(self, flowable, bullet, params=None):
        if params:
            leftIndent = params.get('leftIndent', self._leftIndent)
            rightIndent = params.get('rightIndent', self._rightIndent)
            spaceBefore = params.get('spaceBefore', None)
            spaceAfter = params.get('spaceAfter', None)
            return LIIndenter(flowable, leftIndent, rightIndent, bullet, spaceBefore=spaceBefore, spaceAfter=spaceAfter)
        else:
            return LIIndenter(flowable, self._leftIndent, self._rightIndent, bullet)

    def _makeBullet(self, value, params=None):
        if params is None:

            def getp(a):
                return getattr(self, '_' + a)
        else:
            style = getattr(params, 'style', None)

            def getp(a):
                if a in params:
                    return params[a]
                if style and a in style.__dict__:
                    return getattr(self, a)
                return getattr(self, '_' + a)
        return BulletDrawer(value=value, bulletAlign=getp('bulletAlign'), bulletType=getp('bulletType'), bulletColor=getp('bulletColor'), bulletFontName=getp('bulletFontName'), bulletFontSize=getp('bulletFontSize'), bulletOffsetY=getp('bulletOffsetY'), bulletDedent=getp('calcBulletDedent'), bulletDir=getp('bulletDir'), bulletFormat=getp('bulletFormat'))

    def _getContent(self):
        bt = self._bulletType
        value = self._start
        if isinstance(value, (list, tuple)):
            values = value
            value = values[0]
        else:
            values = [value]
        autov = values[0]
        inc = int(bt in '1aAiI')
        if inc:
            try:
                value = int(value)
            except:
                value = 1
        bd = self._bulletDedent
        if bd == 'auto':
            align = self._bulletAlign
            dir = self._bulletDir
            if dir == 'ltr' and align == 'left':
                bd = self._leftIndent
            elif align == 'right':
                bd = self._rightIndent
            else:
                tvalue = value
                maxW = 0
                for d, f in self._flowablesIter():
                    if d:
                        maxW = max(maxW, _computeBulletWidth(self, tvalue))
                        if inc:
                            tvalue += inc
                    elif isinstance(f, LIIndenter):
                        b = f._bullet
                        if b:
                            if b.bulletType == bt:
                                maxW = max(maxW, _computeBulletWidth(b, b.value))
                                tvalue = int(b.value)
                        else:
                            maxW = max(maxW, _computeBulletWidth(self, tvalue))
                        if inc:
                            tvalue += inc
                if dir == 'ltr':
                    if align == 'right':
                        bd = self._leftIndent - maxW
                    else:
                        bd = self._leftIndent - maxW * 0.5
                elif align == 'left':
                    bd = self._rightIndent - maxW
                else:
                    bd = self._rightIndent - maxW * 0.5
        self._calcBulletDedent = bd
        S = []
        aS = S.append
        i = 0
        for d, f in self._flowablesIter():
            if isinstance(f, ListFlowable):
                fstart = f._start
                if isinstance(fstart, (list, tuple)):
                    fstart = fstart[0]
                if fstart in values:
                    if f._auto:
                        autov = values.index(autov) + 1
                        f._start = values[autov:] + values[:autov]
                        autov = f._start[0]
                        if inc:
                            f._bulletType = autov
                    else:
                        autov = fstart
            fparams = {}
            if not i:
                i += 1
                spaceBefore = getattr(self, 'spaceBefore', None)
                if spaceBefore is not None:
                    fparams['spaceBefore'] = spaceBefore
            if d:
                aS(self._makeLIIndenter(f, bullet=self._makeBullet(value), params=fparams))
                if inc:
                    value += inc
            elif isinstance(f, LIIndenter):
                b = f._bullet
                if b:
                    if b.bulletType != bt:
                        raise ValueError('Included LIIndenter bulletType=%s != OrderedList bulletType=%s' % (b.bulletType, bt))
                    value = int(b.value)
                else:
                    f._bullet = self._makeBullet(value, params=getattr(f, 'params', None))
                if fparams:
                    f.__dict__['spaceBefore'] = max(f.__dict__.get('spaceBefore', 0), spaceBefore)
                aS(f)
                if inc:
                    value += inc
            elif isinstance(f, _LIParams):
                fparams.update(f.params)
                z = self._makeLIIndenter(f.flowable, bullet=None, params=fparams)
                if f.first:
                    if f.value is not None:
                        value = f.value
                        if inc:
                            value = int(value)
                    z._bullet = self._makeBullet(value, f.params)
                    if inc:
                        value += inc
                aS(z)
            else:
                aS(self._makeLIIndenter(f, bullet=None, params=fparams))
        spaceAfter = getattr(self, 'spaceAfter', None)
        if spaceAfter is not None:
            f = S[-1]
            f.__dict__['spaceAfter'] = max(f.__dict__.get('spaceAfter', 0), spaceAfter)
        if self._caption:
            S.insert(0, self._caption)
        return S