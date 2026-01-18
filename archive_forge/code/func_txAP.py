from reportlab.pdfbase.pdfdoc import (PDFObject, PDFArray, PDFDictionary, PDFString, pdfdocEnc,
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import Color, CMYKColor, Whiter, Blacker, opaqueColor
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import isStr, asNative
import weakref
def txAP(self, key, value, iFontName, rFontName, fontSize, shape='square', fillColor=None, borderColor=None, textColor=None, borderWidth=1, borderStyle='solid', width=120, height=36, dashLen=3, wkind='textfield', labels=[], I=[], sel_bg='0.600006 0.756866 0.854904 rg', sel_fg='0 g'):
    stream = [].append
    if opaqueColor(fillColor):
        streamFill = self.streamFillColor(fillColor)
        stream('%(streamFill)s\n0 0 %(width)s %(height)s re\nf')
    if borderWidth != None and borderWidth > 0 and opaqueColor(borderColor):
        hbw = borderWidth * 0.5
        bww = width - borderWidth
        bwh = height - borderWidth
        _2bw = 2 * borderWidth
        if borderStyle in ('bevelled', 'inset'):
            bw2w = width - _2bw
            bw2h = height - _2bw
            if borderStyle == 'bevelled':
                bbs0 = '1 g'
                if fillColor or borderColor:
                    bbs1 = '-0.250977 0.749023 -0.250977 rg'
                else:
                    bbs1 = '.75293 g'
            else:
                bbs0 = '.501953 g'
                bbs1 = '.75293 g'
            stream('%(bbs0)s\n%(borderWidth)s %(borderWidth)s m\n%(borderWidth)s %(bwh)s l\n%(bww)s %(bwh)s l\n%(bw2w)s %(bw2h)s l\n%(_2bw)s %(bw2h)s l\n%(_2bw)s %(_2bw)s l\nf\n%(bbs1)s\n%(bww)s %(bwh)s m\n%(bww)s %(borderWidth)s l\n%(borderWidth)s %(borderWidth)s l\n%(_2bw)s %(_2bw)s l\n%(bw2w)s %(_2bw)s l\n%(bw2w)s %(bw2h)s l\nf')
    else:
        hbw = _2bw = borderWidth = 0
        bww = width
        bwh = height
    undash = ''
    if opaqueColor(borderColor) and borderWidth:
        streamStroke = self.streamStrokeColor(borderColor)
        if borderStyle == 'underlined':
            stream('%(streamStroke)s %(borderWidth)s w 0 %(hbw)s m %(width)s %(hbw)s l s')
        elif borderStyle in ('dashed', 'inset', 'bevelled', 'solid'):
            if borderStyle == 'dashed':
                dash = '\n[%s ] 0 d\n' % fp_str(dashLen)
                undash = '[] 0 d'
            else:
                dash = '\n%s w' % borderWidth
            stream('%(streamStroke)s\n%(dash)s\n%(hbw)s %(hbw)s %(bww)s %(bwh)s re\ns')
    _4bw = 4 * borderWidth
    w4bw = width - _4bw
    h4bw = height - _4bw
    textFill = self.streamFillColor(textColor)
    stream('/Tx BMC \nq\n%(_2bw)s %(_2bw)s %(w4bw)s %(h4bw)s re\nW\nn')
    leading = 1.2 * fontSize
    if wkind == 'listbox':
        nopts = int(h4bw / leading)
        leading = h4bw / float(nopts)
        if nopts > len(labels):
            i0 = 0
            nopts = len(labels)
        elif len(I) <= 1:
            i0 = I[0] if I else 0
            if i0:
                if i0 < nopts:
                    i0 = 0
                else:
                    i = len(labels) - nopts
                    if i0 >= i:
                        i0 = i
        elif I[1] < nopts:
            i0 = 0
        else:
            i0 = I[0]
        y = len(labels)
        i = i0 + nopts
        if i > y:
            i0 = i - y
        ilim = min(y, i0 + nopts)
        if I:
            i = i0
            y = height - _2bw - leading
            stream(sel_bg)
            while i < ilim:
                if i in I:
                    stream('%%(_2bw)s %s %%(w4bw)s %%(leading)s re\nf' % fp_str(y))
                y -= leading
                i += 1
        i = i0
        y = height - _2bw - fontSize
        stream('0 g\n0 G\n%(undash)s')
        while i < ilim:
            stream('BT')
            if i == i0:
                stream('/%(iFontName)s %(fontSize)s Tf')
            stream(sel_fg if i in I else '%(textFill)s')
            stream('%%(_4bw)s %s Td\n(%s) Tj' % (fp_str(y), escPDF(labels[i])))
            y -= leading
            i += 1
            stream('ET')
    else:
        stream('0 g\n0 G\n%(undash)s')
        y = height - fontSize - _2bw
        stream('BT\n/%(iFontName)s %(fontSize)s Tf\n%(textFill)s')
        for line in value.split('\n'):
            stream('%%(_4bw)s %s Td\n(%s) Tj' % (y, escPDF(line)))
            y -= leading
        stream('ET')
    leading = fp_str(leading)
    stream('Q\nEMC\n')
    stream = ('\n'.join(stream.__self__) % vars()).replace('  ', ' ').replace('\n\n', '\n')
    return self.makeStream(width, height, stream, Resources=PDFFromString('<< /ProcSet [/PDF /Text] /Font %(rFontName)s >>' % vars()))