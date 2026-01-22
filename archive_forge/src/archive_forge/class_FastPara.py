from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
class FastPara(Flowable):
    """paragraph with no special features (not even a single ampersand!)"""

    def __init__(self, style, simpletext):
        if '&' in simpletext:
            raise ValueError('no ampersands please!')
        self.style = style
        self.simpletext = simpletext
        self.lines = None

    def wrap(self, availableWidth, availableHeight):
        simpletext = self.simpletext
        self.availableWidth = availableWidth
        style = self.style
        text = self.simpletext
        rightIndent = style.rightIndent
        leftIndent = style.leftIndent
        leading = style.leading
        font = style.fontName
        size = style.fontSize
        firstindent = style.firstLineIndent
        words = simpletext.split()
        lines = []
        from reportlab.pdfbase.pdfmetrics import stringWidth
        spacewidth = stringWidth(' ', font, size)
        currentline = []
        currentlength = 0
        firstmaxlength = availableWidth - rightIndent - firstindent
        maxlength = availableWidth - rightIndent - leftIndent
        if maxlength < spacewidth:
            return (spacewidth + rightIndent + firstindent, availableHeight)
        if availableHeight < leading:
            return (availableWidth, leading)
        if self.lines is None:
            heightused = 0
            cursor = 0
            nwords = len(words)
            done = 0
            while cursor < nwords and (not done):
                thismaxlength = maxlength
                if not lines:
                    thismaxlength = firstmaxlength
                thisword = words[cursor]
                thiswordsize = stringWidth(thisword, font, size)
                if currentlength:
                    thiswordsize = thiswordsize + spacewidth
                nextlength = currentlength + thiswordsize
                if not currentlength or nextlength < maxlength:
                    cursor += 1
                    currentlength = nextlength
                    currentline.append(thisword)
                else:
                    lines.append((' '.join(currentline), currentlength, len(currentline)))
                    currentline = []
                    currentlength = 0
                    heightused = heightused + leading
                    if heightused + leading > availableHeight:
                        done = 1
            if currentlength and (not done):
                lines.append((' '.join(currentline), currentlength, len(currentline)))
                heightused = heightused + leading
            self.lines = lines
            self.height = heightused
            remainder = self.remainder = ' '.join(words[cursor:])
        else:
            remainder = None
            heightused = self.height
            lines = self.lines
        if remainder:
            result = (availableWidth, availableHeight + leading)
        else:
            result = (availableWidth, heightused)
        return result

    def split(self, availableWidth, availableHeight):
        style = self.style
        leading = style.leading
        if availableHeight < leading:
            return []
        lines = self.lines
        if lines is None:
            raise ValueError('must wrap before split')
        remainder = self.remainder
        if remainder:
            next = FastPara(style, remainder)
            return [self, next]
        else:
            return [self]

    def draw(self):
        style = self.style
        lines = self.lines
        rightIndent = style.rightIndent
        leftIndent = style.leftIndent
        leading = style.leading
        font = style.fontName
        size = style.fontSize
        alignment = style.alignment
        firstindent = style.firstLineIndent
        c = self.canv
        escape = c._escape
        height = self.height
        c.translate(0, height - size)
        textobject = c.beginText()
        code = textobject._code
        textobject.setFont(font, size)
        if style.textColor:
            textobject.setFillColor(style.textColor)
        first = 1
        y = 0
        basicWidth = self.availableWidth - rightIndent
        count = 0
        nlines = len(lines)
        while count < nlines:
            text, length, nwords = lines[count]
            count += 1
            thisindent = leftIndent
            if first:
                thisindent = firstindent
            if alignment == TA_LEFT:
                x = thisindent
            elif alignment == TA_CENTER:
                extra = basicWidth - length
                x = thisindent + extra / 2.0
            elif alignment == TA_RIGHT:
                extra = basicWidth - length
                x = thisindent + extra
            elif alignment == TA_JUSTIFY:
                x = thisindent
                if count < nlines and nwords > 1:
                    textobject.setWordSpace((basicWidth - length) / (nwords - 1.0))
                else:
                    textobject.setWordSpace(0.0)
            textobject.setTextOrigin(x, y)
            textobject.textOut(text)
            y = y - leading
        c.drawText(textobject)

    def getSpaceBefore(self):
        return self.style.spaceBefore

    def getSpaceAfter(self):
        return self.style.spaceAfter