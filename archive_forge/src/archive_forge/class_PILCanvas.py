from rdkit.sping.pid import *
import math
import os
class PILCanvas(Canvas):

    def __init__(self, size=(300, 300), name='piddlePIL'):
        self._image = Image.new('RGB', (int(size[0]), int(size[1])), (255, 255, 255))
        self._pen = ImageDraw.ImageDraw(self._image)
        self._setFont(Font())
        Canvas.__init__(self, size, name)

    def __setattr__(self, attribute, value):
        self.__dict__[attribute] = value
        if attribute == 'defaultLineColor':
            self._setColor(self.defaultLineColor)

    def _setColor(self, c):
        """Set the pen color from a piddle color."""
        self._color = (int(c.red * 255), int(c.green * 255), int(c.blue * 255))

    def _setFont(self, font):
        self._font = _pilFont(font)

    def getImage(self):
        return self._image

    def save(self, file=None, format=None):
        """format may be a string specifying a file extension corresponding to
        an image file format. Ex: 'png', 'jpeg', 'gif', 'tif' etc.
        These are defined by PIL, not by us so you need to check the docs.
        In general, I just specify an extension and let format default to None"""
        file = file or self.name
        if hasattr(file, 'write'):
            raise ValueError('fileobj not implemented for piddlePIL')
        if format is None:
            if '.' not in file:
                filename = file + '.png'
            else:
                filename = file
        else:
            filename = file + '.' + format
        self._image.save(filename, format=format)

    def clear(self):
        self.drawRect(0, 0, self.size[0], self.size[1], edgeColor=yellow, fillColor=white)

    def stringWidth(self, s, font=None):
        """Return the logical width of the string if it were drawn         in the current font (defaults to self.defaultFont)."""
        if not font:
            font = self.defaultFont
        if not _widthmaps:
            return font.size * len(s)
        path = _matchingFontPath(font)
        map = _widthmaps[path]
        out = 0
        for c in s:
            out += map.get(c, font.size)
        return out

    def fontAscent(self, font=None):
        """Find the ascent (height above base) of the given font."""
        if not font:
            font = self.defaultFont
        if not _ascents:
            return font.size
        path = _matchingFontPath(font)
        return _ascents[path]

    def fontDescent(self, font=None):
        """Find the descent (extent below base) of the given font."""
        if not font:
            font = self.defaultFont
        if not _descents:
            return font.size / 2
        path = _matchingFontPath(font)
        return _descents[path]

    def drawLine(self, x1, y1, x2, y2, color=None, width=None, dash=None, **kwargs):
        """Draw a straight line between x1,y1 and x2,y2."""
        if width is None:
            w = self.defaultLineWidth
        elif width:
            w = width
        else:
            return
        if color:
            if color == transparent:
                return
            self._setColor(color)
        elif self.defaultLineColor == transparent:
            return
        if not dash:
            self._pen.line((x1, y1, x2, y2), fill=self._color, width=w)
        else:
            dx = x2 - x1
            dy = y2 - y1
            lineLen = math.sqrt(dx * dx + dy * dy)
            theta = math.atan2(dy, dx)
            cosT = math.cos(theta)
            sinT = math.sin(theta)
            pos = (x1, y1)
            dist = 0
            currDash = 0
            dashOn = 1
            while dist < lineLen:
                currL = dash[currDash % len(dash)]
                if dist + currL > lineLen:
                    currL = lineLen - dist
                endP = (pos[0] + currL * cosT, pos[1] + currL * sinT)
                if dashOn:
                    self.drawLine(pos[0], pos[1], endP[0], endP[1], color=color, width=width, dash=None, **kwargs)
                pos = endP
                dist += currL
                currDash += 1
                dashOn = not dashOn

    def drawPolygon(self, pointlist, edgeColor=None, edgeWidth=None, fillColor=None, closed=0, dash=None, **kwargs):
        """drawPolygon(pointlist) -- draws a polygon
        pointlist: a list of (x,y) tuples defining vertices
        """
        pts = list(pointlist)
        for i in range(len(pts)):
            pts[i] = tuple(pts[i])
        filling = 0
        if fillColor:
            if fillColor != transparent:
                self._setColor(fillColor)
                filling = 1
        elif self.defaultFillColor != transparent:
            self._setColor(self.defaultFillColor)
            filling = 1
        if filling:
            pts = [(int(x[0]), int(x[1])) for x in pts]
            self._pen.polygon(pts, fill=self._color)
        if edgeWidth is None:
            edgeWidth = self.defaultLineWidth
        elif not edgeWidth:
            return
        if edgeColor:
            self._setColor(edgeColor)
        else:
            self._setColor(self.defaultLineColor)
        if (closed or (pts[0][0] == pts[-1][0] and pts[0][1] == pts[-1][1])) and edgeWidth <= 1:
            self._pen.polygon(pts, outline=self._color)
        else:
            oldp = pts[0]
            if closed:
                pts.append(oldp)
            for p in pts[1:]:
                self.drawLine(oldp[0], oldp[1], p[0], p[1], edgeColor, edgeWidth, dash=dash, **kwargs)
                oldp = p

    def drawString(self, s, x, y, font=None, color=None, angle=0, **kwargs):
        """Draw a string starting at location x,y."""
        x = int(x)
        y = int(y)
        if '\n' in s or '\r' in s:
            self.drawMultiLineString(s, x, y, font, color, angle, **kwargs)
            return
        if not font:
            font = self.defaultFont
        if not color:
            color = self.defaultLineColor
        if color == transparent:
            return
        sHeight = self.fontAscent(font) + self.fontDescent(font)
        sWidth = self.stringWidth(s, font)
        tempsize = max(sWidth * 1.2, sHeight * 2.0)
        tempimg = Image.new('RGB', (int(tempsize), int(tempsize)), (0, 0, 0))
        temppen = ImageDraw.ImageDraw(tempimg)
        pilfont = _pilFont(font)
        if not pilfont:
            raise ValueError('bad font: %s' % font)
        pos = [4, int(tempsize / 2 - self.fontAscent(font)) - self.fontDescent(font)]
        temppen.text(pos, s, font=pilfont, fill=(255, 255, 255))
        pos[1] = int(tempsize / 2)
        if font.underline:
            ydown = 0.5 * self.fontDescent(font)
            temppen.line((pos[0], pos[1] + ydown, pos[0] + sWidth, pos[1] + ydown))
        if angle:
            from math import cos, pi, sin
            tempimg = tempimg.rotate(angle, Image.BILINEAR)
            temppen = ImageDraw.ImageDraw(tempimg)
            radians = -angle * pi / 180.0
            r = tempsize / 2 - pos[0]
            pos[0] = int(tempsize / 2 - r * cos(radians))
            pos[1] = int(pos[1] - r * sin(radians))
        mask = tempimg.convert('L').point(lambda c: c)
        clr = (int(color.red * 255), int(color.green * 255), int(color.blue * 255))
        temppen.rectangle((0, 0, tempsize, tempsize), fill=clr)
        self._image.paste(tempimg, (int(x) - pos[0], int(y) - pos[1]), mask)

    def drawImage(self, image, x1, y1, x2=None, y2=None, **kwargs):
        """Draw a PIL Image into the specified rectangle.  If x2 and y2 are
        omitted, they are calculated from the image size."""
        if x2 and y2:
            bbox = image.getbbox()
            if x2 - x1 != bbox[2] - bbox[0] or y2 - y1 != bbox[3] - bbox[1]:
                image = image.resize((x2 - x1, y2 - y1))
        self._image.paste(image, (x1, y1))